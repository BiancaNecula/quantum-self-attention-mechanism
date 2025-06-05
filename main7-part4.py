import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split

# Quantum Devices
embed_dim = 4  # Must be power of 2
max_len = 8
n_qubits = int(math.log2(embed_dim))
n_softmax_qubits = int(math.log2(max_len))  # fix: use only log2(max_len) wires
ancilla_qubit = 0
encoding_qubits = list(range(1, n_qubits + 1))
dev = qml.device('default.qubit', wires=[ancilla_qubit] + encoding_qubits)
softmax_dev = qml.device('default.qubit', wires=list(range(n_softmax_qubits)))  # fix: specify actual wire labels

def normalize(v):
    norm = torch.norm(v, p=2)
    return v / (norm + 1e-9)

# Classical Inner Product
def classical_inner_product(q_vec, k_vec):
    return torch.dot(normalize(q_vec), normalize(k_vec))

# Quantum Softmax Wrapper
def quantum_softmax_wrapper(raw_scores):
    raw_scores = F.relu(raw_scores) + 1e-9
    # Apply a sharpening function to emphasize differences
    scaled_scores = raw_scores ** 2
    amplitudes = torch.sqrt(scaled_scores / torch.sum(scaled_scores))
    if len(amplitudes) < max_len:
        amplitudes = F.pad(amplitudes, (0, max_len - len(amplitudes)))
    else:
        amplitudes = amplitudes[:max_len]
    return quantum_softmax(amplitudes)

@qml.qnode(softmax_dev, interface='torch')
def quantum_softmax(amplitudes):
    wires = list(range(n_softmax_qubits))
    state = amplitudes[:2**n_softmax_qubits]
    qml.AmplitudeEmbedding(state, wires=wires, normalize=True)

    # Grover diffusion operator (approximation of partial amplitude amplification)
    state = state / torch.norm(state, p=2)  # Ensure normalization
    qml.templates.MottonenStatePreparation(state, wires=wires)
    qml.templates.GroverOperator(wires=wires)  # A stronger reflection operator

    return qml.probs(wires=wires)

# Quantum Self-Attention Layer
class QuantumSelfAttention(nn.Module):
    def __init__(self, embed_dim, quantum_softmax=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.quantum_softmax = quantum_softmax

    def forward(self, x):
        B, L, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.zeros(B, L, L, device=x.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    score = classical_inner_product(Q[b, i], K[b, j])
                    scores[b, i, j] = score

        if self.quantum_softmax:
            attn = torch.zeros_like(scores)
            for b in range(B):
                for i in range(L):
                    attn[b, i] = quantum_softmax_wrapper(scores[b, i])
        else:
            attn = F.softmax(scores / math.sqrt(D), dim=-1)

        return torch.bmm(attn, V), attn

# Classification Model
class QSAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx, use_quantum_softmax=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.qsa = QuantumSelfAttention(embed_dim, quantum_softmax=use_quantum_softmax)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        context, attn = self.qsa(x)
        pooled = context.mean(dim=1)
        return self.fc(pooled), attn

# Training

def train(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Evaluation

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, y in dataloader:
            output, _ = model(inputs)
            preds.extend(output.argmax(dim=1).tolist())
            labels.extend(y.tolist())
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

# Attention Visualization

def visualize_attention(tokens, attention_weights):
    plt.figure(figsize=(8,8))
    plt.imshow(attention_weights, cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.show()

# Tokenizer

def simple_tokenizer(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()

# Dataset class
class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=8):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = simple_tokenizer(self.texts[idx])
        idxs = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        if len(idxs) < self.max_len:
            idxs += [self.vocab["<pad>"]] * (self.max_len - len(idxs))
        else:
            idxs = idxs[:self.max_len]
        return torch.tensor(idxs), torch.tensor(self.labels[idx])

# Load and prepare data
df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

all_tokens = [token for msg in df['message'] for token in simple_tokenizer(msg)]
vocab_counter = Counter(all_tokens)
vocab = {word: idx+2 for idx, (word, _) in enumerate(vocab_counter.most_common())}
vocab["<pad>"] = 0
vocab["<unk>"] = 1

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
)

train_dataset = SMSDataset(train_texts, train_labels, vocab)
test_dataset = SMSDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
vocab_size = len(vocab)
num_classes = 2
pad_idx = vocab["<pad>"]
# Change this flag to True to enable quantum softmax
use_quantum_softmax = False
model = QSAModel(vocab_size, embed_dim=embed_dim, num_classes=num_classes, pad_idx=pad_idx, use_quantum_softmax=use_quantum_softmax)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Run training and evaluation
train(model, train_loader, criterion, optimizer, epochs=10)
evaluate(model, test_loader)

# Visualize attention
sample_inputs, _ = next(iter(test_loader))
tokens = [list(vocab.keys())[list(vocab.values()).index(i)] if i in vocab.values() else "<unk>" for i in sample_inputs[0].tolist()]
_, attn_weights = model(sample_inputs)
visualize_attention(tokens, attn_weights[0].detach().cpu().numpy())
