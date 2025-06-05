import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import logging
import sys
from datetime import datetime

# --- Global variable declarations ---
vocab = None
VOCAB_SIZE = None
PAD_IDX = None
UNK_IDX = None
NUM_CLASSES = None

# --- Logging Setup ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_filename = f"qsa_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("Initializing Quantum-based Inner-Product with Classical Softmax Model")

# --- Hyperparameters ---
EMBED_DIM = 2           # must be power of 2 for amplitude encoding
MAX_LEN = 20
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
GRAD_CLIP_VALUE = 0.1
N_SHOTS_QIP = 1000
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SMS_DATASET_PATH = "SMSSpamCollection.tsv"

# --- Quantum setup ---
if not (EMBED_DIM > 0 and (EMBED_DIM & (EMBED_DIM - 1)) == 0):
    raise ValueError("EMBED_DIM must be a power of 2.")
N_QUBITS_AE_ENCODING = int(math.log2(EMBED_DIM))
N_QUBITS_ANCILLA = 1

dev_qip = qml.device("default.qubit", wires=N_QUBITS_ANCILLA + N_QUBITS_AE_ENCODING, shots=N_SHOTS_QIP)

@qml.qnode(dev_qip, interface="torch", diff_method="parameter-shift")
def hadamard_test_inner_product(q_features, k_features):
    anc = 0
    enc = list(range(N_QUBITS_ANCILLA, N_QUBITS_ANCILLA + N_QUBITS_AE_ENCODING))

    # Prepare and normalize feature vectors
    def prepare_vec(f):
        norm = torch.linalg.norm(f)
        vec = f
        if not torch.isfinite(norm) or norm < 1e-9:
            vec = torch.ones(EMBED_DIM, device=f.device) / math.sqrt(EMBED_DIM)
        else:
            vec = f / (norm + 1e-9)
        vec = vec * 0.1
        norm2 = torch.linalg.norm(vec)
        if not torch.isfinite(norm2) or norm2 < 1e-9:
            vec = torch.ones(EMBED_DIM, device=f.device) / math.sqrt(EMBED_DIM)
        else:
            vec = vec / (norm2 + 1e-9)
        return vec

    qf = prepare_vec(q_features)
    kf = prepare_vec(k_features)

    # Convert to PennyLane numpy and sanitize
    qf_np = pnp.array(qf.detach().cpu().numpy())
    kf_np = pnp.array(kf.detach().cpu().numpy())
    qf_np = pnp.nan_to_num(qf_np, nan=1.0/math.sqrt(EMBED_DIM), posinf=1.0/math.sqrt(EMBED_DIM), neginf=-1.0/math.sqrt(EMBED_DIM))
    kf_np = pnp.nan_to_num(kf_np, nan=1.0/math.sqrt(EMBED_DIM), posinf=1.0/math.sqrt(EMBED_DIM), neginf=-1.0/math.sqrt(EMBED_DIM))

    # Hadamard test
    qml.Hadamard(wires=anc)
    qml.PauliX(wires=anc)
    qml.ctrl(qml.AmplitudeEmbedding, control=anc)(features=qf_np, wires=enc, normalize=True, pad_with=0.0)
    qml.PauliX(wires=anc)
    qml.ctrl(qml.AmplitudeEmbedding, control=anc)(features=kf_np, wires=enc, normalize=True, pad_with=0.0)
    qml.Hadamard(wires=anc)

    return qml.probs(wires=anc)

# --- Tokenizer & Vocabulary ---
class SimpleTokenizer:
    def __call__(self, text):
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.lower().split()

tokenizer = SimpleTokenizer()

class Vocab:
    def __init__(self, counter, specials, min_freq=1):
        self._stoi = {tok: i for i, tok in enumerate(specials)}
        self._itos = list(specials)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self._stoi:
                self._stoi[token] = len(self._itos)
                self._itos.append(token)
        self.default_index = self._stoi.get(UNK_TOKEN, 0)
    def __call__(self, tokens): return [self._stoi.get(t, self.default_index) for t in tokens]
    def get_stoi(self): return self._stoi
    def get_itos(self): return self._itos
    def __len__(self): return len(self._itos)

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_idx = vocab.get_stoi()[PAD_TOKEN]
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        idxs = self.vocab(tokens)
        if len(idxs) > self.max_len:
            idxs = idxs[:self.max_len]
        else:
            idxs += [self.pad_idx] * (self.max_len - len(idxs))
        return torch.tensor(idxs, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_batch(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

# --- Classical Softmax Self-Attention Layer ---
class QuantumInnerProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.d_k = embed_dim
    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        scores = torch.zeros(B, L, L, device=x.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    probs = hadamard_test_inner_product(x[b,i], x[b,j])
                    z = probs[0] - probs[1]
                    if not torch.isfinite(z):
                        z = 0.0
                    scores[b,i,j] = z * 0.05
        attn = torch.softmax(scores / (math.sqrt(self.d_k) + 1e-9), dim=-1)
        context = torch.bmm(attn, x)
        return context, attn

# --- Classifier ---
class QSAClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)
        with torch.no_grad(): self.embedding.weight[pad_idx].zero_()
        self.attn = QuantumInnerProductAttention(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, text):
        x = self.embedding(text.long())
        x = x * 0.1
        context, attn = self.attn(x)
        pooled = context.mean(dim=1)
        logits = self.fc(pooled)
        return logits, attn

# --- Training & Evaluation ---
def train_and_evaluate(model, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': []}
    for epoch in range(NUM_EPOCHS):
        model.train(); total_loss=0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step(); total_loss += loss.item()
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds==y).sum().item(); total += y.size(0)
        acc = correct/total*100
        logger.info(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Val Acc={acc:.2f}%")
        history['train_loss'].append(total_loss/len(train_loader)); history['val_acc'].append(acc)
    return history

# --- Main ---
if __name__=='__main__':
    df = pd.read_csv(SMS_DATASET_PATH, sep='\t', header=None, names=['label','message'])
    df['label']=df['label'].map({'ham':0,'spam':1})
    df['message']=df['message'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]","", x.lower()))
    texts, labels = df['message'].tolist(), df['label'].tolist()
    counter=Counter()
    for t in texts: counter.update(tokenizer(t))
    vocab = Vocab(counter, specials=[PAD_TOKEN, UNK_TOKEN], min_freq=2)
    VOCAB_SIZE=len(vocab); PAD_IDX=vocab.get_stoi()[PAD_TOKEN]; NUM_CLASSES=2
    t_texts,v_texts,t_labels,v_labels = train_test_split(texts, labels,test_size=0.2,random_state=42,stratify=labels)
    train_ds=TextDataset(t_texts,t_labels,vocab,tokenizer,MAX_LEN)
    val_ds=TextDataset(v_texts,v_labels,vocab,tokenizer,MAX_LEN)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_batch)
    val_loader  =DataLoader(val_ds,batch_size=BATCH_SIZE,collate_fn=collate_batch)
    model = QSAClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, PAD_IDX)
    history = train_and_evaluate(model, train_loader, val_loader)
    logger.info("Training complete.")
     
