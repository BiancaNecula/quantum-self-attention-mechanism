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

# --- Configuratie si Hiperparametri ---
N_QUBITS_SHI = 4   # Numarul de qubiti pentru modelul Shi-QSAN
MAX_LEN = 8        # Lungimea maxima a secventei
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-4 # Rata de invatare redusa pentru stabilitate

# --- Setup Dispozitive Cuantice ---
shi_qsan_dev = qml.device('default.qubit', wires=N_QUBITS_SHI)


# =================================================================================
# MODEL DIN LUCRAREA QSAN de Shi et al. (Logic-Based)
# =================================================================================

class ShiQSANAttention(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        weight_shapes = {"weights": (n_qubits,)}

        # Definim circuitul cuantic ca o functie locala
        def U_M_circuit(inputs, weights):
            """Circuit pentru proiectiile U_q, U_k, U_v din lucrarea lui Shi."""
            # Codare in baza computationala (basis encoding)
            qml.BasisState(inputs.int(), wires=range(self.n_qubits))
            
            # Ansamblu variabil simplificat, inspirat de lucrare
            # FIX: Aplicam Hadamard pe fiecare qubit individual
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            for i in range(self.n_qubits):
                qml.RY(weights[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
            # Extragem un vector clasic de valori asteptate
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Cream un QNode din functia circuitului
        U_M_qnode = qml.QNode(U_M_circuit, shi_qsan_dev, interface="torch")

        # Cream straturi TorchLayer folosind QNode-ul corect
        self.q_layer = qml.qnn.TorchLayer(U_M_qnode, weight_shapes)
        self.k_layer = qml.qnn.TorchLayer(U_M_qnode, weight_shapes)
        self.v_layer = qml.qnn.TorchLayer(U_M_qnode, weight_shapes)

    def forward(self, x_ids):
        B, L = x_ids.shape
        x_basis = torch.zeros(B, L, self.n_qubits, device=x_ids.device)
        for b in range(B):
            for l in range(L):
                token_id = x_ids[b, l].item()
                # Asiguram ca ID-ul tokenului nu depaseste capacitatea de codare
                if token_id >= 2**self.n_qubits:
                    token_id = (2**self.n_qubits) - 1 # Plafonam la valoarea maxima
                binary_rep = format(token_id, f'0{self.n_qubits}b')
                x_basis[b, l] = torch.tensor([int(bit) for bit in binary_rep])
        
        # Initializam tensori pentru rezultate
        Q = torch.zeros(B, L, self.n_qubits, device=x_ids.device)
        K = torch.zeros(B, L, self.n_qubits, device=x_ids.device)
        V = torch.zeros(B, L, self.n_qubits, device=x_ids.device)

        # Procesam fiecare element individual pentru a evita eroarea de batch
        for b in range(B):
            for l in range(L):
                Q[b, l] = self.q_layer(x_basis[b, l])
                K[b, l] = self.k_layer(x_basis[b, l])
                V[b, l] = self.v_layer(x_basis[b, l])

        scores = torch.zeros(B, L, L, device=x_ids.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    # Simplificare QLS: folosim cosinus similarity
                    scores[b, i, j] = F.cosine_similarity(Q[b, i], K[b, j], dim=0)

        attn_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, V), attn_weights

class ShiQSANModel(nn.Module):
    def __init__(self, n_qubits, num_classes):
        super().__init__()
        # Acest model nu are un strat de embedding, deoarece opereaza direct pe ID-uri
        self.attention = ShiQSANAttention(n_qubits)
        self.fc = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        context, _ = self.attention(x)
        return self.fc(context.mean(dim=1)), _

# =================================================================================
# LOGICA PENTRU DATASET, ANTRENARE SI EVALUARE
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
    print(f"Pornire antrenament pentru {model.__class__.__name__}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoca {epoch+1}/{epochs}, Pierdere: {avg_loss:.4f}")

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
    print(f"Rezultate evaluare pentru {model.__class__.__name__}:")
    print(f"  Acuratete: {accuracy:.4f}")
    print(f"  Scor F1: {f1:.4f}\n")
    return accuracy

# =================================================================================
# SCRIPT PRINCIPAL DE EXECUTIE
# =================================================================================
if __name__ == "__main__":
    print("Incarcare si pregatire set de date SMS Spam...")
    try:
        df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None, names=["label", "message"])
    except FileNotFoundError:
        print("Eroare: SMSSpamCollection.tsv nu a fost gasit.")
        exit()

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    all_tokens = [token for msg in df['message'] for token in simple_tokenizer(msg)]
    vocab = {word: i+2 for i, (word, _) in enumerate(Counter(all_tokens).most_common())}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    VOCAB_SIZE = len(vocab)
    NUM_CLASSES = 2

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
    )
    train_dataset = SMSDataset(train_texts, train_labels, vocab, MAX_LEN)
    test_dataset = SMSDataset(test_texts, test_labels, vocab, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    # Model QSAN de Shi (Logic-Based)
    model_shi = ShiQSANModel(N_QUBITS_SHI, NUM_CLASSES)
    optimizer_shi = torch.optim.Adam(model_shi.parameters(), lr=LEARNING_RATE)
    train_model(model_shi, train_loader, criterion, optimizer_shi, EPOCHS)
    accuracy_shi = evaluate_model(model_shi, test_loader)
    
    print("="*40)
    print("      REZULTAT FINAL")
    print("="*40)
    print(f"Acuratete Shi-QSAN Model: {accuracy_shi:.4f}")
    print("="*40)
