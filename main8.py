import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Configuration and Hyperparameters ---
# We'll use consistent settings for a fair comparison
EMBED_DIM = 4      # Dimension of token embeddings.
MAX_LEN = 8        # Max sequence length for padding/truncating.
N_QUBITS_QSANN = 4 # Number of qubits for the QSANN model.
BATCH_SIZE = 16
EPOCHS = 10
# Reduced learning rate for better stability with quantum layers
LEARNING_RATE = 5e-4

# --- Quantum Device Setup ---
# Device for the thesis model's quantum inner product (Hadamard Test)
n_data_qubits = int(math.log2(EMBED_DIM))
inner_product_dev = qml.device('default.qubit', wires=1 + n_data_qubits) # 1 ancilla

# Device for the thesis model's quantum softmax (PAA)
softmax_dev = qml.device('default.qubit', wires=int(math.log2(MAX_LEN)))

# Device for the QSANN model's quantum layers
qsann_dev = qml.device('default.qubit', wires=N_QUBITS_QSANN)


# =================================================================================
# 1. MODEL FROM YOUR THESIS (Fully Quantum Attention)
# =================================================================================
# This model now uses a quantum inner product (Hadamard Test) and a 
# quantum softmax (Partial Amplitude Amplification).

# Using 'parameter-shift' as it's more robust for complex operations
@qml.qnode(inner_product_dev, interface='torch', diff_method='parameter-shift')
def quantum_inner_product_circuit(q_vec, k_vec):
    """
    Calculates the real part of the inner product <k|q> using a Hadamard test.
    This circuit measures Re(<0| U_k^dagger U_q |0>).
    """
    ancilla_wire = 0
    data_wires = list(range(1, 1 + n_data_qubits))

    # Define state preparation operators and let PennyLane handle normalization
    # This is more numerically stable than manual normalization.
    U_q = qml.AmplitudeEmbedding(q_vec, wires=data_wires, normalize=True)
    U_k_dagger = qml.adjoint(qml.AmplitudeEmbedding(k_vec, wires=data_wires, normalize=True))

    # Hadamard test circuit
    qml.Hadamard(wires=ancilla_wire)
    
    # Apply controlled operations sequentially to avoid decomposition issues
    # with composite operators like qml.prod.
    qml.ctrl(U_q, control=ancilla_wire)
    qml.ctrl(U_k_dagger, control=ancilla_wire)

    qml.Hadamard(wires=ancilla_wire)
    
    return qml.expval(qml.PauliZ(ancilla_wire))


@qml.qnode(softmax_dev, interface='torch', diff_method='backprop')
def quantum_softmax_paa_circuit(amplitudes, oracle_diag, rounds):
    """Implements Partial Amplitude Amplification to mimic softmax."""
    n_softmax_qubits = len(softmax_dev.wires)
    wires = list(range(n_softmax_qubits))

    # Grover diffusion operator reflection term (reflection about |0...0>)
    reflection_diag = -torch.ones(2**n_softmax_qubits)
    reflection_diag[0] = 1.0
    
    # 1. State preparation from initial amplitudes
    qml.AmplitudeEmbedding(amplitudes, wires=wires, normalize=True)

    # 2. PAA iterations (Oracle + Diffusion)
    for _ in range(rounds):
        # Oracle marks states with amplitudes above a threshold
        qml.DiagonalQubitUnitary(oracle_diag, wires=wires)
        
        # Diffusion operator reflects about the average amplitude
        qml.adjoint(qml.AmplitudeEmbedding(amplitudes, wires=wires, normalize=True))
        qml.DiagonalQubitUnitary(reflection_diag.to(amplitudes.device), wires=wires)
        qml.AmplitudeEmbedding(amplitudes, wires=wires, normalize=True)

    return qml.probs(wires=wires)

def quantum_softmax_paa_wrapper(raw_scores, rounds=1, percentile=75):
    """Prepares scores and calls the PAA circuit."""
    # Step 1: Clip and shift scores to be non-negative
    processed_scores = F.relu(raw_scores) + 1e-9

    # Step 2: Build amplitudes for state preparation
    squared_scores = processed_scores**2
    total_squared = torch.sum(squared_scores)
    amplitudes = torch.sqrt(squared_scores / (total_squared + 1e-9))

    # Pad or truncate to the number of basis states
    target_len = 2**len(softmax_dev.wires)
    if amplitudes.shape[0] < target_len:
        padding = (0, target_len - amplitudes.shape[0])
        amplitudes = F.pad(amplitudes, padding)
    elif amplitudes.shape[0] > target_len:
        amplitudes = amplitudes[:target_len]

    # Step 3: Define the soft oracle based on a percentile threshold
    # The oracle flips the phase of states corresponding to high scores
    if torch.all(amplitudes < 1e-9):
        oracle_diag = torch.ones(target_len, device=raw_scores.device)
    else:
        # Use only non-zero amplitudes to compute the threshold
        non_zero_amps = amplitudes[amplitudes > 1e-9]
        if non_zero_amps.numel() > 0:
            threshold = torch.quantile(non_zero_amps, percentile / 100.0)
        else:
            threshold = 0.0 # Fallback if all amplitudes are effectively zero
            
        oracle_diag = torch.tensor(
            [-1.0 if amp >= threshold else 1.0 for amp in amplitudes],
            device=raw_scores.device
        )

    return quantum_softmax_paa_circuit(amplitudes, oracle_diag, rounds)


class ThesisAttention(nn.Module):
    """Fully quantum self-attention based on the Thesis."""
    def __init__(self, embed_dim, paa_rounds=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.paa_rounds = paa_rounds
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.zeros(B, L, L, device=x.device)
        # Calculate similarity scores using Quantum Inner Product
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    q_vec = Q[b, i]
                    k_vec = K[b, j]
                    
                    # FIX: Detach tensors from the PyTorch computation graph before
                    # passing them to the QNode. This prevents autograd conflicts.
                    if torch.norm(q_vec) < 1e-9 or torch.norm(k_vec) < 1e-9:
                        scores[b, i, j] = 0.0
                    else:
                        scores[b, i, j] = quantum_inner_product_circuit(q_vec.detach(), k_vec.detach())

        # Apply quantum softmax (PAA) to get attention weights
        attn_weights = torch.zeros_like(scores)
        for b in range(B):
            for i in range(L):
                attn_weights[b, i] = quantum_softmax_paa_wrapper(scores[b, i], rounds=self.paa_rounds)
        
        # Apply attention weights to Value vectors
        output = torch.bmm(attn_weights, V)
        return output, attn_weights

class ThesisQSAModel(nn.Module):
    """The complete model architecture as described in the thesis."""
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = ThesisAttention(embed_dim, paa_rounds=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x_embed = self.embedding(x)
        context, attn_weights = self.attention(x_embed)
        pooled = context.mean(dim=1)
        return self.fc(pooled), attn_weights

# =================================================================================
# 2. MODEL FROM THE PAPER (QSANN - Gaussian Projected Quantum Self-Attention)
# =================================================================================
# This model uses quantum circuits to replace the Q, K, V linear projections.
# Attention scores are calculated based on the distance between 1D projections
# of quantum states.

def qsann_ansatz(weights):
    """A simple ansatz for the QSANN model's Q,K,V circuits."""
    qml.BasicEntanglerLayers(weights=weights, wires=range(N_QUBITS_QSANN))

def create_qsann_qnode(n_qubits, n_layers):
    """Creates a QNode for QSANN's Q, K, or V projections."""
    shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    
    @qml.qnode(qsann_dev, interface='torch', diff_method='backprop')
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qsann_ansatz(weights)
        return qml.expval(qml.PauliZ(0))
        
    return qnode, shape

class QSANNAttention(nn.Module):
    """Self-Attention layer based on the QSANN paper."""
    def __init__(self, embed_dim, n_qubits):
        super().__init__()
        self.embed_dim = embed_dim
        n_layers = 2
        
        q_node, q_shape = create_qsann_qnode(n_qubits, n_layers)
        self.q_layer = qml.qnn.TorchLayer(q_node, weight_shapes={"weights": q_shape})

        k_node, k_shape = create_qsann_qnode(n_qubits, n_layers)
        self.k_layer = qml.qnn.TorchLayer(k_node, weight_shapes={"weights": k_shape})
        
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)

        z_q = self.q_layer(x_flat).reshape(B, L)
        z_k = self.k_layer(x_flat).reshape(B, L)
        
        V = self.v_proj(x)

        scores = torch.zeros(B, L, L, device=x.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    scores[b, i, j] = torch.exp(-(z_q[b, i] - z_k[b, j])**2)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, V)
        return output, attn_weights

class QSANNModel(nn.Module):
    """The complete QSANN model architecture from the paper."""
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx, n_qubits):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = QSANNAttention(embed_dim, n_qubits)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x_embed = self.embedding(x)
        context, attn_weights = self.attention(x_embed)
        pooled = context.mean(dim=1)
        return self.fc(pooled), attn_weights

# =================================================================================
# 3. DATASET AND TRAINING/EVALUATION LOGIC
# =================================================================================

def simple_tokenizer(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()

class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token_id = vocab["<pad>"]
        self.unk_token_id = vocab["<unk>"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = simple_tokenizer(self.texts[idx])
        token_ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.pad_token_id] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
            
        return torch.tensor(token_ids), torch.tensor(self.labels[idx])

def train_model(model, dataloader, criterion, optimizer, epochs):
    """Generic training loop with gradient clipping for stability."""
    print(f"Starting training for {model.__class__.__name__}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs, _ = model(inputs)
            preds = outputs.argmax(dim=1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Evaluation Results for {model.__class__.__name__}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}\n")
    return accuracy

# =================================================================================
# 4. MAIN EXECUTION SCRIPT
# =================================================================================

if __name__ == "__main__":
    print("Loading and preparing SMS Spam dataset...")
    try:
        df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None, names=["label", "message"])
    except FileNotFoundError:
        print("Error: SMSSpamCollection.tsv not found.")
        print("Please make sure the dataset file is in the same directory as the script.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit()

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    all_tokens = [token for msg in df['message'] for token in simple_tokenizer(msg)]
    vocab_counter = Counter(all_tokens)
    vocab = {word: i+2 for i, (word, _) in enumerate(vocab_counter.most_common())}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab["<pad>"]
    NUM_CLASSES = 2

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
    )

    train_dataset = SMSDataset(train_texts, train_labels, vocab, MAX_LEN)
    test_dataset = SMSDataset(test_texts, test_labels, vocab, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    results = {}

    # 1. Thesis Model
    model_thesis = ThesisQSAModel(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, PAD_IDX)
    optimizer_thesis = torch.optim.Adam(model_thesis.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_model(model_thesis, train_loader, criterion, optimizer_thesis, epochs=EPOCHS)
    accuracy_thesis = evaluate_model(model_thesis, test_loader)
    results['Thesis Model'] = accuracy_thesis

    # 2. QSANN Model
    model_qsann = QSANNModel(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, PAD_IDX, N_QUBITS_QSANN)
    optimizer_qsann = torch.optim.Adam(model_qsann.parameters(), lr=LEARNING_RATE)
    
    train_model(model_qsann, train_loader, criterion, optimizer_qsann, epochs=EPOCHS)
    accuracy_qsann = evaluate_model(model_qsann, test_loader)
    results['QSANN Model'] = accuracy_qsann
    
    # --- Final Comparison ---
    print("="*30)
    print("      FINAL ACCURACY COMPARISON")
    print("="*30)
    for model_name, acc in results.items():
        print(f"{model_name:<20}: {acc:.4f}")
    print("="*30)
