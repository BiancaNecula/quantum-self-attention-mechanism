import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset, TensorDataset
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

# ─── Logging & Reproducibility ─────────────────────────────────────────────────
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
log_filename = f"qsa_training_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
torch.manual_seed(42)
pnp.random.seed(42)
logger.info("Initializing Quantum Self-Attention (classical softmax only)")

# ─── Hyperparameters & Paths ────────────────────────────────────────────────────
EMBED_DIM     = 2      # must be a power of two
MAX_LEN       = 20
BATCH_SIZE    = 2
LEARNING_RATE = 1e-3   # increased for stronger learning
NUM_EPOCHS    = 5
GRAD_CLIP     = 0.1
PAD_TOKEN     = "<pad>"
UNK_TOKEN     = "<unk>"
SMS_PATH      = "SMSSpamCollection.tsv"

# ─── Quantum Inner-Product Device & QNode ───────────────────────────────────────
if EMBED_DIM & (EMBED_DIM - 1) != 0:
    raise ValueError("EMBED_DIM must be a power of 2")
N_QUBITS_AE      = int(math.log2(EMBED_DIM))
N_QUBITS_ANCILLA = 1
dev_qip = qml.device(
    "default.qubit",
    wires=N_QUBITS_ANCILLA + N_QUBITS_AE,
    shots=None           # analytic mode, no sampling noise
)

@qml.qnode(dev_qip, interface="torch", diff_method="parameter-shift")
def hadamard_test_re_inner_product_circuit(qf, kf):
    anc = 0
    enc = list(range(N_QUBITS_ANCILLA, N_QUBITS_ANCILLA + N_QUBITS_AE))

    def prep(v):
        n = torch.linalg.norm(v)
        if not torch.isfinite(n) or n < 1e-9:
            return torch.ones(EMBED_DIM, device=v.device) / math.sqrt(EMBED_DIM)
        return v / (n + 1e-9)

    qv = prep(qf) * 0.1
    kv = prep(kf) * 0.1
    # clamp NaN/Inf
    qv = torch.nan_to_num(qv, nan=1/math.sqrt(EMBED_DIM), posinf=1/math.sqrt(EMBED_DIM), neginf=-1/math.sqrt(EMBED_DIM))
    kv = torch.nan_to_num(kv, nan=1/math.sqrt(EMBED_DIM), posinf=1/math.sqrt(EMBED_DIM), neginf=-1/math.sqrt(EMBED_DIM))

    # Hadamard-test circuit
    qml.Hadamard(wires=anc)
    qml.PauliX(wires=anc)
    qml.ctrl(qml.AmplitudeEmbedding, control=anc)(
        features=qv.detach().cpu().numpy(),
        wires=enc, normalize=True, pad_with=0.0
    )
    qml.PauliX(wires=anc)
    qml.ctrl(qml.AmplitudeEmbedding, control=anc)(
        features=kv.detach().cpu().numpy(),
        wires=enc, normalize=True, pad_with=0.0
    )
    qml.Hadamard(wires=anc)
    return qml.probs(wires=anc)


# ─── Tokenizer & Vocabulary ─────────────────────────────────────────────────────
class SimpleTokenizer:
    def __call__(self, txt):
        txt = re.sub(r"[^A-Za-z0-9\s]", "", txt)
        return txt.lower().split()

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

    def get_stoi(self): return self._stoi
    def get_itos(self): return self._itos
    def __len__(self): return len(self._itos)


# ─── Dataset & Collate ─────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.vocab, self.tokenizer = vocab, tokenizer
        self.max_len = max_len
        self.pad_idx = vocab.get_stoi()[PAD_TOKEN]

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


# ─── Quantum-Enhanced Self-Attention Layer ──────────────────────────────────────
class QuantumSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.d_k = embed_dim
        self.sqrt_d_k = math.sqrt(self.d_k)

        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.v_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

        # learnable temperature for softmax
        self.temp = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.05)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.05)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.05)

    def forward(self, x_embed, pad_mask):
        B, L, _ = x_embed.shape

        # Project, norm, dropout, clamp
        q = torch.nan_to_num(self.dropout(self.q_norm(self.W_q(x_embed) * 0.05)),
                             nan=0.0, posinf=1e-6, neginf=-1e-6)
        k = torch.nan_to_num(self.dropout(self.k_norm(self.W_k(x_embed) * 0.05)),
                             nan=0.0, posinf=1e-6, neginf=-1e-6)
        v = torch.nan_to_num(self.dropout(self.v_norm(self.W_v(x_embed) * 0.05)),
                             nan=0.0, posinf=1e-6, neginf=-1e-6)

        # Compute raw scores via quantum inner product
        scores = torch.zeros(B, L, L, device=x_embed.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    try:
                        p = hadamard_test_re_inner_product_circuit(q[b, i], k[b, j])
                        z = p[0] - p[1]
                        if not torch.isfinite(z):
                            z = 0.0
                    except Exception as e:
                        logger.error(f"Hadamard error b={b},i={i},j={j}: {e}")
                        z = 0.0
                    # no extra scaling here
                    scores[b, i, j] = z

        # Mask out padding positions (–1e9 → softmax→0)
        mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        scores = scores.masked_fill(mask, float(-1e9))

        # Classical softmax with learned temperature
        attn = nn.functional.softmax(scores * self.temp / self.sqrt_d_k, dim=-1)

        # Context vectors (no extra scaling)
        context = torch.bmm(attn, v)
        return context, attn


# ─── Final Classifier ──────────────────────────────────────────────────────────
class QSATextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.005)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()
        self.embed_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout    = nn.Dropout(0.1)
        self.pad_idx    = pad_idx

        self.qsa = QuantumSelfAttentionLayer(embed_dim)
        self.fc  = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.05)

    def forward(self, text_ids):
        # Embedding + norm + dropout + clamp
        x = self.embedding(text_ids) * 0.1
        x = torch.nan_to_num(self.embed_norm(x), nan=0.0, posinf=1e-6, neginf=-1e-6)
        x = self.dropout(x)

        # Build pad mask
        pad_mask = (text_ids == self.pad_idx)

        # Self-attention
        context, attn = self.qsa(x, pad_mask)

        # Pool & final layer
        pooled = context.mean(dim=1).clamp(-1.0, 1.0) * 0.1
        logits = self.fc(pooled).clamp(-1.0, 1.0)
        return logits, attn


# ─── Training & Evaluation ─────────────────────────────────────────────────────
def train_and_evaluate_model(model, train_loader, val_loader, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "val_prec": [],  "val_rec":  [], "val_f1":  [],
        "t_train":   [], "t_val":    []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        t0, train_loss = time.time(), 0.0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
        t_train = time.time() - t0
        avg_train = train_loss / len(train_loader)

        t1, val_loss, preds, labs = time.time(), 0.0, [], []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                val_loss += criterion(logits, y).item()
                p = logits.argmax(dim=1)
                preds += p.cpu().tolist()
                labs  += y.cpu().tolist()
        t_val   = time.time() - t1
        avg_val = val_loss / len(val_loader)

        acc  = accuracy_score(labs, preds) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(
            labs, preds, average="weighted", zero_division=0
        )

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(acc)
        history["val_prec"].append(prec)
        history["val_rec"].append(rec)
        history["val_f1"].append(f1)
        history["t_train"].append(t_train)
        history["t_val"].append(t_val)

        scheduler.step(avg_val)

        logger.info(
            f"Epoch {epoch}: TrainLoss={avg_train:.4f}, ValLoss={avg_val:.4f}, "
            f"Acc={acc:.2f}%, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, "
            f"Ttrain={t_train:.2f}s, Tval={t_val:.2f}s"
        )

    # ─── Plot metrics ────────────────────────────────────────────────────────────
    es = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(es, history["train_loss"], "bo-", label="Train Loss")
    plt.plot(es, history["val_loss"],   "ro-", label="Val Loss")
    plt.title("Loss"), plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(es, history["val_acc"], "go-")
    plt.title("Accuracy")

    plt.subplot(2, 2, 3)
    plt.plot(es, history["val_prec"], "co-", label="Precision")
    plt.plot(es, history["val_rec"],  "mo-", label="Recall")
    plt.plot(es, history["val_f1"],   "yo-", label="F1")
    plt.title("Prec/Rec/F1"), plt.legend()

    plt.tight_layout()
    plt.savefig("metrics.png")
    logger.info("Saved metrics.png")

    # ─── Attention map (first val batch) ─────────────────────────────────────────
    xb, _ = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        _, attn = model(xb.to(device))

    tokens = [vocab.get_itos()[i] for i in xb[0].tolist()]
    M = attn[0].cpu().numpy()

    plt.figure(figsize=(6, 6))
    im = plt.imshow(M, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
    plt.yticks(range(len(tokens)), tokens, fontsize=6)
    plt.colorbar(im)
    plt.title("Attention Map (first sample)")
    plt.tight_layout()
    plt.savefig("attention_map.png")
    logger.info("Saved attention_map.png")

    return history


# ─── Main Execution ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load & preprocess data
    df = pd.read_csv(SMS_PATH, sep="\t", header=None, names=["label", "message"])
    df["label"]   = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].apply(
        lambda x: re.sub(r"[^A-Za-z0-9\s]", "", x.lower())
    )

    texts = df["message"].tolist()
    labs   = df["label"].tolist()
    ctr    = Counter(tok for t in texts for tok in tokenizer(t))

    vocab   = Vocab(ctr, specials=[PAD_TOKEN, UNK_TOKEN], min_freq=2)
    PAD_IDX = vocab.get_stoi()[PAD_TOKEN]

    Xtr, Xva, ytr, yva = train_test_split(
        texts, labs, test_size=0.2, random_state=42, stratify=labs
    )

    tr_ds = TextDataset(Xtr, ytr, vocab, tokenizer, MAX_LEN)
    va_ds = TextDataset(Xva, yva, vocab, tokenizer, MAX_LEN)

    tr_loader = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    va_loader = DataLoader(
        va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    model   = QSATextClassifier(len(vocab), EMBED_DIM, 2, PAD_IDX)
    history = train_and_evaluate_model(model, tr_loader, va_loader, vocab)
    logger.info("Training complete.")
