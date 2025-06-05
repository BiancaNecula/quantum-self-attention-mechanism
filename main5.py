import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import logging
import sys
from datetime import datetime

# --- Logging Setup ---
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
log_filename = f"qsa_training_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("Initializing Quantum Inner-Product + Classical Softmax Model")

# --- Hyperparameters ---
EMBED_DIM       = 2    # must be a power of 2
MAX_LEN         = 20
BATCH_SIZE      = 2
LEARNING_RATE   = 5e-5
NUM_EPOCHS      = 5
GRAD_CLIP_VALUE = 0.1
N_SHOTS_QIP     = 1000
PAD_TOKEN       = "<pad>"
UNK_TOKEN       = "<unk>"
SMS_DATA_PATH   = "SMSSpamCollection.tsv"

# --- Quantum setup (inner-product only) ---
if EMBED_DIM & (EMBED_DIM - 1) != 0:
    raise ValueError("EMBED_DIM must be a power of 2")
N_QUBITS_AE      = int(math.log2(EMBED_DIM))
N_QUBITS_ANCILLA = 1
dev_qip = qml.device(
    "default.qubit",
    wires=N_QUBITS_ANCILLA + N_QUBITS_AE,
    shots=N_SHOTS_QIP
)

@qml.qnode(dev_qip, interface="torch", diff_method="parameter-shift")
def hadamard_test_inner_product(qf, kf):
    anc = 0
    enc = list(range(N_QUBITS_ANCILLA, N_QUBITS_ANCILLA + N_QUBITS_AE))

    def prep(f):
        n = torch.linalg.norm(f)
        if not torch.isfinite(n) or n < 1e-9:
            return torch.ones(EMBED_DIM, device=f.device) / math.sqrt(EMBED_DIM)
        return f / (n + 1e-9)

    qf = prep(qf) * 0.1
    kf = prep(kf) * 0.1

    # clamp any stray NaNs/Infs
    qf = torch.nan_to_num(qf,
                          nan=1.0/math.sqrt(EMBED_DIM),
                          posinf=1.0/math.sqrt(EMBED_DIM),
                          neginf=-1.0/math.sqrt(EMBED_DIM))
    kf = torch.nan_to_num(kf,
                          nan=1.0/math.sqrt(EMBED_DIM),
                          posinf=1.0/math.sqrt(EMBED_DIM),
                          neginf=-1.0/math.sqrt(EMBED_DIM))

    # Hadamard-test circuit
    qml.Hadamard(wires=anc)
    qml.PauliX(wires=anc)
    qml.ctrl(qml.AmplitudeEmbedding, control=anc)(
        features=qf.detach().cpu().numpy(),
        wires=enc, normalize=True, pad_with=0.0
    )
    qml.PauliX(wires=anc)
    qml.ctrl(qml.AmplitudeEmbedding, control=anc)(
        features=kf.detach().cpu().numpy(),
        wires=enc, normalize=True, pad_with=0.0
    )
    qml.Hadamard(wires=anc)
    return qml.probs(wires=anc)

# --- Tokenizer & Vocab ---
class SimpleTokenizer:
    def __call__(self, text):
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)
        return text.lower().split()

tokenizer = SimpleTokenizer()

class Vocab:
    def __init__(self, counter, specials, min_freq=1):
        self._stoi = {tok: i for i, tok in enumerate(specials)}
        self._itos = list(specials)
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self._stoi:
                self._stoi[tok] = len(self._itos)
                self._itos.append(tok)
        self.default = self._stoi.get(UNK_TOKEN, 0)

    def __call__(self, tokens):
        return [self._stoi.get(t, self.default) for t in tokens]

    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        return self._itos

    def __len__(self):
        return len(self._itos)

# --- Dataset & Collate ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len):
        self.texts      = texts
        self.labels     = labels
        self.vocab      = vocab
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.pad_idx    = vocab.get_stoi()[PAD_TOKEN]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer(self.texts[idx])
        idxs = self.vocab(toks)
        if len(idxs) > self.max_len:
            idxs = idxs[:self.max_len]
        else:
            idxs += [self.pad_idx] * (self.max_len - len(idxs))
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_batch(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

# --- Self-Attention Layer (classical softmax) ---
class QuantumSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_shots_qip):
        super().__init__()
        self.embed_dim   = embed_dim
        self.d_k         = embed_dim
        self.sqrt_d_k    = math.sqrt(self.d_k)

        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, self.d_k, bias=False)
        self.W_k = nn.Linear(embed_dim, self.d_k, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.q_norm = nn.LayerNorm(self.d_k, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.d_k, eps=1e-6)
        self.v_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.05)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.05)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.05)

    def forward(self, x_embed, pad_mask):
        B, L, _ = x_embed.shape

        # project + norm + dropout
        q = self.dropout(self.q_norm(self.W_q(x_embed) * 0.05))
        k = self.dropout(self.k_norm(self.W_k(x_embed) * 0.05))
        v = self.dropout(self.v_norm(self.W_v(x_embed) * 0.05))

        # clamp any stray NaNs/Infs
        q = torch.nan_to_num(q, nan=0.0, posinf=1e-6, neginf=-1e-6)
        k = torch.nan_to_num(k, nan=0.0, posinf=1e-6, neginf=-1e-6)
        v = torch.nan_to_num(v, nan=0.0, posinf=1e-6, neginf=-1e-6)

        # compute pairwise scores via Hadamard test
        scores = torch.zeros(B, L, L, device=x_embed.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    try:
                        p = hadamard_test_inner_product(q[b,i], k[b,j])
                        z = p[0] - p[1]
                        if not torch.isfinite(z):
                            z = 0.0
                    except Exception as e:
                        logger.error(f"Hadamard test error b{b} i{i} j{j}: {e}")
                        z = 0.0
                    # **no more 0.05 squeeze** here:
                    scores[b,i,j] = z

        # **mask out pads** before softmax
        # pad_mask: (B,L) True at pad positions
        mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        scores = scores.masked_fill(mask, float(-1e9))

        # classical softmax with √d scaling
        attn = nn.functional.softmax(scores / self.sqrt_d_k, dim=-1)

        # context vectors
        context = torch.bmm(attn, v) * 0.05
        return context, attn

# --- Classifier ---
class QSATextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.005)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

        self.embed_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout    = nn.Dropout(0.1)
        self.pad_idx    = pad_idx

        self.qsa = QuantumSelfAttentionLayer(embed_dim, N_SHOTS_QIP)
        self.fc  = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.05)

    def forward(self, text_ids):
        # embedding + norm + dropout
        x = self.embedding(text_ids) * 0.1
        x = self.embed_norm(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e-6, neginf=-1e-6)
        x = self.dropout(x)

        # build pad mask
        pad_mask = (text_ids == self.pad_idx)

        # self-attention
        context, attn = self.qsa(x, pad_mask)

        # pooling + final
        pooled = context.mean(dim=1).clamp(-1.0, 1.0) * 0.1
        logits = self.fc(pooled).clamp(-1.0, 1.0)
        return logits, attn

# --- Training & Evaluation (with full metrics + plots) ---
def train_and_evaluate(model, train_loader, val_loader, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=2
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'val_prec':   [], 'val_rec':   [], 'val_f1':   [],
        't_train':    [], 't_val':     []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step()
            epoch_loss += loss.item()
        t_train = time.time() - t0
        avg_train_loss = epoch_loss / len(train_loader)

        t1 = time.time()
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
        t_val = time.time() - t1
        avg_val_loss = val_loss / len(val_loader)

        acc  = accuracy_score(all_labels, all_preds) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        # record
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        history['val_prec'].append(prec)
        history['val_rec'].append(rec)
        history['val_f1'].append(f1)
        history['t_train'].append(t_train)
        history['t_val'].append(t_val)

        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
            f"Acc={acc:.2f}%, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, "
            f"T_train={t_train:.2f}s, T_val={t_val:.2f}s"
        )

    # ——— Plot metrics ———
    ep = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(ep, history['train_loss'], 'bo-', label='Train Loss')
    plt.plot(ep, history['val_loss'],   'ro-', label='Val Loss')
    plt.title('Loss'), plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(ep, history['val_acc'], 'go-', label='Val Acc')
    plt.title('Accuracy'), plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(ep, history['val_prec'], 'co-', label='Precision')
    plt.plot(ep, history['val_rec'],  'mo-', label='Recall')
    plt.plot(ep, history['val_f1'],   'yo-', label='F1')
    plt.title('Prec/Rec/F1'), plt.legend()

    plt.tight_layout()
    plt.savefig("metrics2.png")
    logger.info("Saved metrics2.png")

    # ——— Attention map for first validation batch ———
    xb, _ = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        _, attn = model(xb.to(device))

    tokens = [vocab.get_itos()[i] for i in xb[0].tolist()]
    mat = attn[0].cpu().numpy()

    plt.figure(figsize=(6, 6))
    im = plt.imshow(mat, cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
    plt.yticks(range(len(tokens)), tokens, fontsize=6)
    plt.colorbar(im)
    plt.title("Attention Map (first sample)")
    plt.tight_layout()
    plt.savefig("attention_map2.png")
    logger.info("Saved attention_map2.png")

    return history

if __name__ == "__main__":
    # ——— Load & preprocess ———
    df = pd.read_csv(SMS_DATA_PATH, sep='\t', header=None, names=['label','message'])
    df['label']   = df['label'].map({'ham':0,'spam':1})
    df['message'] = df['message'].apply(lambda x: re.sub(r"[^A-Za-z0-9\s]","", x.lower()))

    texts  = df['message'].tolist()
    labels = df['label'].tolist()
    ctr    = Counter(token for t in texts for token in tokenizer(t))

    vocab   = Vocab(ctr, specials=[PAD_TOKEN, UNK_TOKEN], min_freq=2)
    PAD_IDX = vocab.get_stoi()[PAD_TOKEN]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = TextDataset(X_train, y_train, vocab, tokenizer, MAX_LEN)
    val_ds   = TextDataset(X_val,   y_val,   vocab, tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE,             collate_fn=collate_batch
    )

    model   = QSATextClassifier(len(vocab), EMBED_DIM, 2, PAD_IDX)
    history = train_and_evaluate(model, train_loader, val_loader, vocab)
    logger.info("Training complete.")
