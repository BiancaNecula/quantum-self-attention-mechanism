# ================================================================
#  Hybrid Quantum-Classical SMS-Spam classifier
#  – Hadamard-test similarity  +  Amplitude-amplified soft-max
#  – Four attention modes, selectable via flags
# ================================================================

import math, time, re
from collections import Counter
import pennylane as qml
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
#  FLAGS – choose the flavour you want
# ────────────────────────────────────────────────────────────────
USE_QUANTUM_INNER_PRODUCT = True      # Hadamard-test vs. cosine
USE_QUANTUM_SOFTMAX      = True       # Grover-soft-max vs. classical

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
embed_dim, max_len = 4, 8               # both must be powers of 2
dtype = torch.float64; torch.set_default_dtype(dtype)

# ----------------------------------------------------------------
#  Quantum devices
# ----------------------------------------------------------------
n_q_data     = int(math.log2(embed_dim))
n_q_softmax  = int(math.log2(max_len))

softmax_dev  = qml.device("default.qubit", wires=range(n_q_softmax), shots=None)
if USE_QUANTUM_INNER_PRODUCT:
    ip_dev = qml.device("default.qubit",
                        wires=range(1 + 2 * n_q_data),  # ancilla + two regs
                        shots=None)

# ----------------------------------------------------------------
#  Helper: make safe amplitudes (||ψ|| = 1, never 0)
# ----------------------------------------------------------------
def amplitude(vec: torch.Tensor, n_qubits: int):
    norm = torch.linalg.vector_norm(vec)
    if norm < 1e-6:
        state = torch.zeros(2**n_qubits, dtype=dtype); state[0] = 1.0
    else:
        state = vec / norm
    return state

# ----------------------------------------------------------------
#  Hadamard-test inner product   (|⟨q|k⟩|²)
# ----------------------------------------------------------------
def classical_cosine(q, k):
    return F.cosine_similarity(q.unsqueeze(0), k.unsqueeze(0)).squeeze(0)

if USE_QUANTUM_INNER_PRODUCT:
    anc     = 0
    wires_A = range(1, 1 + n_q_data)
    wires_B = range(1 + n_q_data, 1 + 2 * n_q_data)

    @qml.qnode(ip_dev, interface="torch", diff_method="backprop")
    def hadamard_test(a_state, b_state):
        # |0>|q>|k>
        qml.AmplitudeEmbedding(a_state, wires=wires_A, normalize=False)
        qml.AmplitudeEmbedding(b_state, wires=wires_B, normalize=False)
        # H on ancilla, then CSWAP, then H, measure Z
        qml.Hadamard(wires=anc)
        for wa, wb in zip(wires_A, wires_B):
            qml.CSWAP(wires=[anc, wa, wb])
        qml.Hadamard(wires=anc)
        return qml.expval(qml.PauliZ(anc))

    def quantum_inner(q_vec, k_vec):
        q_amp = amplitude(q_vec, n_q_data)
        k_amp = amplitude(k_vec, n_q_data)
        return hadamard_test(q_amp, k_amp)          # already differentiable

# ----------------------------------------------------------------
#  Grover-style quantum soft-max  (soft oracle + 3 rounds)
# ----------------------------------------------------------------
@qml.qnode(softmax_dev, interface="torch", diff_method="backprop")
def grover_softmax(amps):
    amps = amplitude(amps, n_q_softmax)             # safe normalisation
    wires = range(n_q_softmax)
    qml.templates.MottonenStatePreparation(amps, wires=wires)

    # soft oracle: phase-flip amplitudes ≥ 75-th percentile
    thr = torch.quantile(amps, 0.75).detach().cpu().item()
    oracle_diag = [-1.0 if a >= thr else 1.0 for a in amps]
    for _ in range(3):                              # partial amplification
        qml.DiagonalQubitUnitary(oracle_diag, wires=wires)
        qml.templates.GroverOperator(wires=wires)

    return qml.probs(wires=wires)

def quantum_softmax_wrapper(raw_scores: torch.Tensor):
    # make non-negative & emphasise differences
    shifted = F.relu(raw_scores) + 1e-6
    probs   = shifted.pow(2)
    probs   = probs / probs.sum() + 1e-12
    amps    = torch.sqrt(probs)
    amps    = F.pad(amps, (0, max_len - amps.numel())) if amps.numel() < max_len else amps[:max_len]
    return grover_softmax(amps.double())

# ----------------------------------------------------------------
#  Attention layer
# ----------------------------------------------------------------
class QSelfAttn(nn.Module):
    def __init__(self, D, q_inner=True, q_soft=True):
        super().__init__()
        self.q_inner, self.q_soft = q_inner, q_soft
        self.q_proj = nn.Linear(D, D, False)
        self.k_proj = nn.Linear(D, D, False)
        self.v_proj = nn.Linear(D, D, False)

    def _sim(self, q, k):
        if self.q_inner:
            return quantum_inner(q, k)
        return classical_cosine(q, k)

    def forward(self, x):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        B, L, _ = Q.shape

        if self.q_inner:
            scores = torch.zeros(B, L, L, dtype=x.dtype)
            for b in range(B):
                for i in range(L):
                    for j in range(L):
                        scores[b, i, j] = self._sim(Q[b, i], K[b, j])
        else:
            scores = torch.bmm(F.normalize(Q, dim=-1),
                               F.normalize(K, dim=-1).transpose(1, 2))

        if self.q_soft:
            attn = torch.stack([
                torch.stack([quantum_softmax_wrapper(scores[b, i])
                             for i in range(L)]) for b in range(B)
            ])[..., :L]
        else:
            attn = F.softmax(scores / math.sqrt(Q.size(-1)), dim=-1)

        return torch.bmm(attn, V), attn

# ----------------------------------------------------------------
#  Classifier
# ----------------------------------------------------------------
class QSAModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, D=4, C=2,
                 q_inner=True, q_soft=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D, padding_idx=pad_idx)
        self.qsa = QSelfAttn(D, q_inner, q_soft)
        self.fc  = nn.Linear(D, C)

    def forward(self, tok):
        x = self.emb(tok).double()
        ctx, att = self.qsa(x)
        return self.fc(ctx.mean(1)), att

# ----------------------------------------------------------------
#  Train / eval helpers (unchanged)
# ----------------------------------------------------------------
def eval_metrics(model, loader):
    model.eval(); pred, gold = [], []
    with torch.no_grad():
        for x, y in loader:
            pred.extend(model(x)[0].argmax(1).tolist())
            gold.extend(y.tolist())
    a = accuracy_score(gold, pred)
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, average="weighted", zero_division=0)
    return a, p, r, f1

def train(model, tr, vl, crit, opt, epochs=5):
    for ep in range(1, epochs+1):
        model.train(); run = 0.
        it = pb(tr, desc=f"Epoch {ep}/{epochs}", leave=False)
        for i, (x, y) in enumerate(it, 1):
            opt.zero_grad(); loss = crit(model(x)[0], y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item()
            if _tqdm: it.set_postfix(loss=run/i)
        a, p, r, f1 = eval_metrics(model, vl)
        print(f"[{time.strftime('%H:%M:%S')}] Ep {ep} │ "
              f"loss {run/len(tr):.4f} │ Acc {a:.3f} P {p:.3f} R {r:.3f} F1 {f1:.3f}")

# ----------------------------------------------------------------
#  Data prep (unchanged)
# ----------------------------------------------------------------
def tok(s): return re.sub(r"[^a-z0-9\s]", "", s.lower()).split()

class SMS(Dataset):
    def __init__(self, txt, lbl, vocab, L):
        self.txt, self.lbl, self.v, self.L = txt, lbl, vocab, L
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
train_dl = DataLoader(SMS(tr_txt, tr_lbl, vocab, max_len), batch_size=16, shuffle=True)
test_dl  = DataLoader(SMS(te_txt, te_lbl, vocab, max_len), batch_size=16)

# ----------------------------------------------------------------
#  Run
# ----------------------------------------------------------------
print(f"\n>>> {('quantum','classical')[not USE_QUANTUM_INNER_PRODUCT]} inner-product + "
      f"{('quantum','classical')[not USE_QUANTUM_SOFTMAX]} soft-max <<<\n")

model = QSAModel(len(vocab), vocab["<pad>"],
                 embed_dim, 2,
                 q_inner=USE_QUANTUM_INNER_PRODUCT,
                 q_soft=USE_QUANTUM_SOFTMAX)
opt  = torch.optim.Adam(model.parameters(), lr=3e-4)
crit = nn.CrossEntropyLoss()

train(model, train_dl, test_dl, crit, opt, epochs=10)
a, p, r, f1 = eval_metrics(model, test_dl)
print(f"\nFINAL │ Acc {a:.3f}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")

# ----------------------------------------------------------------
#  (Optional) attention heat-map
# ----------------------------------------------------------------
def viz(tok_list, att):
    plt.figure(figsize=(8,8)); plt.imshow(att, cmap="viridis")
    plt.xticks(range(len(tok_list)), tok_list, rotation=90)
    plt.yticks(range(len(tok_list)), tok_list); plt.colorbar(); plt.tight_layout(); plt.show()

id2w = {v:k for k,v in vocab.items()}
sample_x, _ = next(iter(test_dl))
tok_list = [id2w.get(i.item(), "<unk>") for i in sample_x[0]]
_, Att = model(sample_x)
viz(tok_list, Att[0].detach().numpy())
