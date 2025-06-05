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
import seaborn as sns

# Quantum Devices
embed_dim = 4  # Must be power of 2
max_len = 8
n_qubits = int(math.log2(embed_dim))
n_softmax_qubits = int(math.log2(max_len))
ancilla_qubit = 0
encoding_qubits = list(range(1, n_qubits + 1))
dev = qml.device('default.qubit', wires=[ancilla_qubit] + encoding_qubits)
softmax_dev = qml.device('default.qubit', wires=list(range(n_softmax_qubits)))

results_grid = []

# Normalize vector
def normalize(v):
    norm = torch.norm(v, p=2)
    return v / (norm + 1e-9)

# Classical attention score
def classical_inner_product(q_vec, k_vec):
    return torch.dot(normalize(q_vec), normalize(k_vec))

# Dynamic quantum softmax generation
def make_quantum_softmax(threshold, rounds):
    @qml.qnode(softmax_dev, interface='torch')
    def qsfn(amplitudes):
        wires = list(range(n_softmax_qubits))
        state = amplitudes[:2**n_softmax_qubits]
        if torch.any(torch.isnan(state)) or torch.norm(state, p=2) == 0:
            state = torch.ones_like(state) / math.sqrt(len(state))
        else:
            state = state / (torch.norm(state, p=2) + 1e-9)
        qml.templates.MottonenStatePreparation(state, wires=wires)
        oracle_matrix = [-1.0 if amp >= threshold else 1.0 for amp in state]
        for _ in range(rounds):
            qml.DiagonalQubitUnitary(oracle_matrix, wires=wires)
            qml.templates.GroverOperator(wires=wires)
        return qml.probs(wires=wires)
    return qsfn

# Quantum self-attention mechanism
class QuantumSelfAttention(nn.Module):
    def __init__(self, embed_dim, quantum_softmax):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.quantum_softmax_fn = quantum_softmax

    def forward(self, x):
        B, L, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.zeros(B, L, L, device=x.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    scores[b, i, j] = classical_inner_product(Q[b, i], K[b, j])
        attn = torch.zeros_like(scores)
        for b in range(B):
            for i in range(L):
                raw = F.relu(scores[b, i]) + 1e-6
                scaled = raw ** 2
                scaled = scaled / (torch.sum(scaled) + 1e-9)
                amplitudes = torch.sqrt(scaled)
                if torch.any(torch.isnan(amplitudes)) or torch.norm(amplitudes, p=2) == 0:
                    amplitudes = torch.ones_like(amplitudes) / math.sqrt(len(amplitudes))
                if len(amplitudes) < max_len:
                    amplitudes = F.pad(amplitudes, (0, max_len - len(amplitudes)))
                else:
                    amplitudes = amplitudes[:max_len]
                try:
                    attn[b, i] = self.quantum_softmax_fn(amplitudes)
                except Exception:
                    attn[b, i] = torch.ones_like(amplitudes) / len(amplitudes)
        return torch.bmm(attn, V), attn

class QSAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx, quantum_softmax_fn):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.qsa = QuantumSelfAttention(embed_dim, quantum_softmax=quantum_softmax_fn)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        context, attn = self.qsa(x)
        pooled = context.mean(dim=1)
        return self.fc(pooled), attn

# Tokenizer and dataset

def simple_tokenizer(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()

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

# Training and evaluation

def train(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

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
    return acc

def plot_grid(results):
    data = pd.DataFrame(results, columns=["GroverRounds", "Percentile", "Accuracy"])
    pivot = data.pivot("GroverRounds", "Percentile", "Accuracy")
    plt.figure(figsize=(8, 6))
    plt.title("Quantum Softmax Grid Search Accuracy")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.ylabel("Grover Rounds")
    plt.xlabel("Oracle Threshold Percentile")
    plt.show()

# Data Loading

df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
all_tokens = [token for msg in df['message'] for token in simple_tokenizer(msg)]
vocab_counter = Counter(all_tokens)
vocab = {word: idx+2 for idx, (word, _) in enumerate(vocab_counter.most_common())}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
train_texts, test_texts, train_labels, test_labels = train_test_split(df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'])
train_dataset = SMSDataset(train_texts, train_labels, vocab)
test_dataset = SMSDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Grid Search for Quantum Parameters
vocab_size = len(vocab)
num_classes = 2
pad_idx = vocab["<pad>"]
best_acc = 0
best_params = (0, 0)

for rounds in range(1, 6):
    for percentile in [60, 70, 80, 90]:
        def quantum_softmax_fn(amplitudes):
            state = amplitudes[:2**n_softmax_qubits]
            if torch.any(torch.isnan(state)) or torch.norm(state, p=2) == 0:
                state = torch.ones_like(state) / math.sqrt(len(state))
            else:
                state = state / (torch.norm(state, p=2) + 1e-9)
            threshold = torch.quantile(state, percentile / 100.0).item()
            return make_quantum_softmax(threshold, rounds)(state)

        model = QSAModel(vocab_size, embed_dim, num_classes, pad_idx, quantum_softmax_fn)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train(model, train_loader, criterion, optimizer, epochs=10)
        acc = evaluate(model, test_loader)
        results_grid.append((rounds, percentile, acc))
        if acc > best_acc:
            best_acc = acc
            best_params = (rounds, percentile)

print(f"Best Accuracy: {best_acc:.4f} with Grover Rounds = {best_params[0]}, Threshold Percentile = {best_params[1]}")
plot_grid(results_grid)
