# ================================================================
#  Hybrid Quantum-Classical SMS-Spam classifier
#  – NaN-safe quantum softmax
#  – compatible with older PennyLane (no dtype= kwarg)
# ================================================================

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

# ----------------------------------------------------------------
#  GLOBAL SETTINGS
# ----------------------------------------------------------------
embed_dim = 4           # must be 2^k
max_len   = 8           # sequence length (2^k)
dtype     = torch.float64
torch.set_default_dtype(dtype)

# ----------------------------------------------------------------
#  Quantum devices  (no dtype= kwarg for older PL versions)
# ----------------------------------------------------------------
n_qubits         = int(math.log2(embed_dim))
n_softmax_qubits = int(math.log2(max_len))

dev = qml.device("default.qubit", wires=n_qubits, shots=None)
softmax_dev = qml.device("default.qubit", wires=list(range(n_softmax_qubits)), shots=None)

# ----------------------------------------------------------------
#  Helper
# ----------------------------------------------------------------
def classical_inner_product(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(q.unsqueeze(0), k.unsqueeze(0)).squeeze(0)

# ----------------------------------------------------------------
#  Differentiable quantum softmax
# ----------------------------------------------------------------
@qml.qnode(softmax_dev, interface="torch", diff_method="backprop")
def quantum_softmax(amplitudes: torch.Tensor) -> torch.Tensor:
    amps = amplitudes / (torch.linalg.vector_norm(amplitudes) + 1e-12)
    qml.AmplitudeEmbedding(amps, wires=range(n_softmax_qubits), normalize=False)
    return qml.probs(wires=range(n_softmax_qubits))

def quantum_softmax_wrapper(raw_scores: torch.Tensor) -> torch.Tensor:
    shifted = raw_scores - raw_scores.min()
    shifted = shifted + 1e-6                            # avoid zeros
    probs   = shifted.pow(2)
    probs   = probs / probs.sum() + 1e-12
    amps    = torch.sqrt(probs)

    # pad / truncate to register size
    amps = F.pad(amps, (0, max_len - amps.numel())) if amps.numel() < max_len else amps[:max_len]
    return quantum_softmax(amps.double())

# ----------------------------------------------------------------
#  Quantum self-attention
# ----------------------------------------------------------------
class QuantumSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, use_q_softmax: bool = True):
        super().__init__()
        self.use_q_softmax = use_q_softmax
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores  = torch.bmm(F.normalize(Q, dim=-1),
                            F.normalize(K, dim=-1).transpose(1, 2))

        if self.use_q_softmax:
            attn = torch.stack([
                torch.stack([quantum_softmax_wrapper(scores[b, i])
                             for i in range(scores.size(1))])
                for b in range(scores.size(0))
            ])
            attn = attn[..., :scores.size(-1)]
        else:
            attn = F.softmax(scores / math.sqrt(Q.size(-1)), dim=-1)

        context = torch.bmm(attn, V)
        return context, attn

# ----------------------------------------------------------------
#  Classifier
# ----------------------------------------------------------------
class QSAModel(nn.Module):
    def __init__(self, vocab_size, pad_idx,
                 embed_dim=4, num_classes=2, use_q_softmax=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.qsa       = QuantumSelfAttention(embed_dim, use_q_softmax)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        x = self.embedding(tokens).double()         # (B,L,D)
        ctx, attn = self.qsa(x)
        pooled = ctx.mean(dim=1)                    # (B,D)
        return self.fc(pooled), attn

# ----------------------------------------------------------------
#  Train / evaluate
# ----------------------------------------------------------------
def train(model, loader, criterion, optim, epochs=5):
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optim.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()
        print(f"Epoch {ep+1}: loss {total/len(loader):.4f}")

def evaluate(model, loader):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for x, y in loader:
            out, _ = model(x)
            preds.extend(out.argmax(1).tolist())
            gold.extend(y.tolist())
    acc = accuracy_score(gold, preds)
    p,r,f1,_ = precision_recall_fscore_support(gold, preds, average="weighted", zero_division=0)
    print(f"Acc {acc:.3f} | Prec {p:.3f} | Rec {r:.3f} | F1 {f1:.3f}")

# ----------------------------------------------------------------
#  Tokeniser & dataset
# ----------------------------------------------------------------
def tok(text: str):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts, self.labels = texts, labels
        self.vocab, self.max_len = vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tok(self.texts[i])][:self.max_len]
        ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[i])

# ----------------------------------------------------------------
#  Load data
# ----------------------------------------------------------------
df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None,
                 names=["label", "msg"])
df["label"] = df["label"].map({"ham":0, "spam":1})

# ---- FIXED: build vocab from **flattened** token list ------------
all_tokens = [t for m in df.msg for t in tok(m)]
vocab = {w:i+2 for i,(w,_) in enumerate(Counter(all_tokens).most_common())}
vocab["<pad>"], vocab["<unk>"] = 0, 1
# -----------------------------------------------------------------

tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
    df.msg.tolist(), df.label.tolist(),
    test_size=0.2, random_state=42, stratify=df.label
)

train_ds = SMSDataset(tr_txt, tr_lbl, vocab, max_len)
test_ds  = SMSDataset(te_txt, te_lbl, vocab, max_len)
train_ld = DataLoader(train_ds, batch_size=16, shuffle=True)
test_ld  = DataLoader(test_ds, batch_size=16)

# ----------------------------------------------------------------
#  Run
# ----------------------------------------------------------------
model = QSAModel(len(vocab), vocab["<pad>"], embed_dim, 2, True)
optim = torch.optim.Adam(model.parameters(), lr=3e-4)
crit  = nn.CrossEntropyLoss()

train(model, train_ld, crit, optim, 10)
evaluate(model, test_ld)

# ----------------------------------------------------------------
#  (Optional) attention visualisation
# ----------------------------------------------------------------
def viz(tokens, attn):
    plt.figure(figsize=(8,8))
    plt.imshow(attn, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar(); plt.tight_layout(); plt.show()

id2word = {v:k for k,v in vocab.items()}
batch_x, _ = next(iter(test_ld))
sample_tokens = [id2word.get(idx.item(), "<unk>") for idx in batch_x[0]]
_, att = model(batch_x)
viz(sample_tokens, att[0].detach().numpy())
