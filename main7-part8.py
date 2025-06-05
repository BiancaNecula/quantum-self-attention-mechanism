# ================================================================
#  Hybrid Quantum-Classical SMS-Spam classifier
#  – four attention modes, selectable by flags
#  – progress logs, NaN-safe
# ================================================================

import math, time, re
from collections import Counter
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
#  FLAGS – pick the variant you want
# ────────────────────────────────────────────────────────────────
USE_QUANTUM_INNER_PRODUCT = True     # ⇠ set False for classical cosine sim
USE_QUANTUM_SOFTMAX      = False     # ⇠ set False for classical softmax

# ----------------------------------------------------------------
#  tqdm progress bar (optional)
# ----------------------------------------------------------------
try:
    from tqdm.auto import tqdm
    pb, _tqdm = lambda x, **kw: tqdm(x, **kw), True
except ImportError:
    print("tqdm not installed → using plain prints.")
    pb, _tqdm = lambda x, **kw: x, False

# ----------------------------------------------------------------
#  GLOBAL HYPERPARAMS
# ----------------------------------------------------------------
embed_dim, max_len = 4, 8            # both must be 2^k
dtype = torch.float64; torch.set_default_dtype(dtype)

# ----------------------------------------------------------------
#  Quantum devices
# ----------------------------------------------------------------
n_qubits_data    = int(math.log2(embed_dim))
n_qubits_softmax = int(math.log2(max_len))
softmax_dev = qml.device("default.qubit", wires=range(n_qubits_softmax), shots=None)

if USE_QUANTUM_INNER_PRODUCT:
    ip_total_wires = 1 + 2 * n_qubits_data          # ancilla + two registers
    ip_dev = qml.device("default.qubit", wires=range(ip_total_wires), shots=None)

# ----------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------
def classical_cosine(q_vec, k_vec):
    return F.cosine_similarity(q_vec.unsqueeze(0), k_vec.unsqueeze(0)).squeeze(0)

# ★A —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def make_amplitude(vec: torch.Tensor):
    """Return a valid amplitude vector (norm-1)."""
    norm = torch.linalg.vector_norm(vec)
    if norm < 1e-6:
        state = torch.zeros(2**n_qubits_data, dtype=dtype)
        state[0] = 1.0                       # |00…0⟩
    else:
        state = vec.clone() / norm
    return state
# ★C —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if USE_QUANTUM_INNER_PRODUCT:
    ancilla   = 0
    wires_A   = range(1, 1 + n_qubits_data)
    wires_B   = range(1 + n_qubits_data, 1 + 2 * n_qubits_data)

    @qml.qnode(ip_dev, interface="torch", diff_method="backprop")
    def swap_test(a_state, b_state):
        qml.AmplitudeEmbedding(a_state, wires=wires_A, normalize=False)
        qml.AmplitudeEmbedding(b_state, wires=wires_B, normalize=False)
        qml.Hadamard(wires=ancilla)
        for wa, wb in zip(wires_A, wires_B):
            qml.CSWAP(wires=[ancilla, wa, wb])
        qml.Hadamard(wires=ancilla)
        return qml.expval(qml.PauliZ(ancilla))

    def quantum_inner_product(q_vec, k_vec):
        # ★D — use safe amplitudes
        q_amp = make_amplitude(q_vec)
        k_amp = make_amplitude(k_vec)
        return swap_test(q_amp, k_amp)       # scalar tensor
# ----------------------------------------------------------------
@qml.qnode(softmax_dev, interface="torch", diff_method="backprop")
def quantum_softmax_circ(amps):
    amps = amps / (torch.linalg.vector_norm(amps) + 1e-12)
    qml.AmplitudeEmbedding(amps, wires=range(n_qubits_softmax), normalize=False)
    return qml.probs(wires=range(n_qubits_softmax))

def quantum_softmax_wrapper(raw_scores):
    shifted = raw_scores - raw_scores.min() + 1e-6
    probs   = shifted.pow(2); probs = probs / probs.sum() + 1e-12
    amps    = torch.sqrt(probs)
    amps    = F.pad(amps, (0, max_len - amps.numel())) if amps.numel() < max_len else amps[:max_len]
    return quantum_softmax_circ(amps.double())

# ----------------------------------------------------------------
#  Attention layer
# ----------------------------------------------------------------
class QuantumSelfAttention(nn.Module):
    def __init__(self, D, q_inner=True, q_soft=True):
        super().__init__()
        self.q_inner, self.q_soft = q_inner, q_soft
        self.q_proj = nn.Linear(D, D, False)
        self.k_proj = nn.Linear(D, D, False)
        self.v_proj = nn.Linear(D, D, False)

    def _sim(self, q, k):
        return quantum_inner_product(q, k) if self.q_inner else classical_cosine(q, k)

    def forward(self, x):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        B, L, _ = Q.shape
        # similarity matrix
        if self.q_inner:
            scores = torch.zeros(B, L, L, dtype=x.dtype)
            for b in range(B):
                for i in range(L):
                    for j in range(L):
                        scores[b, i, j] = self._sim(Q[b, i], K[b, j])
        else:
            scores = torch.bmm(F.normalize(Q, dim=-1),
                               F.normalize(K, dim=-1).transpose(1, 2))
        # softmax
        if self.q_soft:
            attn = torch.stack([
                torch.stack([quantum_softmax_wrapper(scores[b, i])
                             for i in range(L)]) for b in range(B)
            ])[..., :L]
        else:
            attn = F.softmax(scores / math.sqrt(Q.size(-1)), dim=-1)
        return torch.bmm(attn, V), attn

# ----------------------------------------------------------------
#  Model
# ----------------------------------------------------------------
class QSAModel(nn.Module):
    def __init__(self, vocab, pad_idx, D=4, C=2,
                 q_inner=True, q_soft=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, D, padding_idx=pad_idx)
        self.qsa = QuantumSelfAttention(D, q_inner, q_soft)
        self.fc  = nn.Linear(D, C)

    def forward(self, tok):
        x = self.embedding(tok).double()
        ctx, attn = self.qsa(x)
        return self.fc(ctx.mean(1)), attn

# ----------------------------------------------------------------
#  Train / eval helpers
# ----------------------------------------------------------------
def eval_metrics(model, loader):
    model.eval(); p, g = [], []
    with torch.no_grad():
        for x, y in loader:
            p.extend(model(x)[0].argmax(1).tolist())
            g.extend(y.tolist())
    a = accuracy_score(g, p)
    pr, rc, f1, _ = precision_recall_fscore_support(g, p, average="weighted", zero_division=0)
    return a, pr, rc, f1

def train(model, tr, vl, crit, opt, epochs=5):
    for ep in range(1, epochs+1):
        model.train(); run = 0.
        it = pb(tr, desc=f"Epoch {ep}/{epochs}", leave=False)
        for i, (x, y) in enumerate(it, 1):
            opt.zero_grad(); out = model(x)[0]; loss = crit(out, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += loss.item(); (it.set_postfix if _tqdm else lambda **_: None)(loss=run/i)
        a, pr, rc, f1 = eval_metrics(model, vl)
        print(f"[{time.strftime('%H:%M:%S')}] Ep {ep} │ "
              f"loss {run/len(tr):.4f} │ Acc {a:.3f} P {pr:.3f} R {rc:.3f} F1 {f1:.3f}")

# ----------------------------------------------------------------
#  Tokeniser / data
# ----------------------------------------------------------------
def tok(s): return re.sub(r"[^a-z0-9\s]", "", s.lower()).split()

class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, L):
        self.txt, self.lbl, self.v, self.L = texts, labels, vocab, L
    def __len__(self): return len(self.txt)
    def __getitem__(self, i):
        ids = [self.v.get(t, self.v["<unk>"]) for t in tok(self.txt[i])][:self.L]
        ids += [self.v["<pad>"]] * (self.L - len(ids))
        return torch.tensor(ids), torch.tensor(self.lbl[i])

df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None, names=["lab","msg"])
df.lab = df.lab.map({"ham":0,"spam":1})
vocab_cnt = Counter(t for m in df.msg for t in tok(m))
vocab = {w:i+2 for i,(w,_) in enumerate(vocab_cnt.most_common())}; vocab["<pad>"]=0; vocab["<unk>"]=1
tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(df.msg.tolist(), df.lab.tolist(),
                                                 test_size=0.2, stratify=df.lab, random_state=42)
train_dl = DataLoader(SMSDataset(tr_txt, tr_lbl, vocab, max_len), batch_size=16, shuffle=True)
test_dl  = DataLoader(SMSDataset(te_txt, te_lbl, vocab, max_len), batch_size=16)

# ----------------------------------------------------------------
#  Run
# ----------------------------------------------------------------
print(f"\n>>> {('quantum','classical')[not USE_QUANTUM_INNER_PRODUCT]} inner-product + "
      f"{('quantum','classical')[not USE_QUANTUM_SOFTMAX]} softmax <<<\n")

model = QSAModel(len(vocab), vocab["<pad>"],
                 embed_dim, 2,
                 q_inner=USE_QUANTUM_INNER_PRODUCT,
                 q_soft=USE_QUANTUM_SOFTMAX)
opt  = torch.optim.Adam(model.parameters(), lr=3e-4)
crit = nn.CrossEntropyLoss()

train(model, train_dl, test_dl, crit, opt, epochs=10)
a, pr, rc, f1 = eval_metrics(model, test_dl)
print(f"\nFINAL │ Acc {a:.3f} P {pr:.3f} R {rc:.3f} F1 {f1:.3f}")

# ----------------------------------------------------------------
#  (Optional) attention heat-map
# ----------------------------------------------------------------
def viz(tok_list, att):
    plt.figure(figsize=(8,8)); plt.imshow(att, cmap="viridis")
    plt.xticks(range(len(tok_list)), tok_list, rotation=90)
    plt.yticks(range(len(tok_list)), tok_list); plt.colorbar(); plt.tight_layout(); plt.show()

id2w = {v:k for k,v in vocab.items()}
sample_x, _ = next(iter(test_dl))
tokens = [id2w.get(i.item(), "<unk>") for i in sample_x[0]]
_, A = model(sample_x)
viz(tokens, A[0].detach().numpy())
