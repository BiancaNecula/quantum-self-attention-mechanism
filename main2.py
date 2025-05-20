import pennylane as qml
from pennylane import numpy as pnp 
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd # For loading CSV/TSV
import re # For basic text cleaning
from collections import Counter
from sklearn.model_selection import train_test_split # For splitting dataset
import logging # For improved logging
import sys
from datetime import datetime

# --- Configuration class to manage global state ---
class Config:
    def __init__(self):
        self.vocab = None
        self.VOCAB_SIZE = None
        self.PAD_IDX = None
        self.UNK_IDX = None
        self.NUM_CLASSES = None

# Create a single instance of Config to be used throughout the code
config = Config()

# --- Module-level variables ---
# These variables are used throughout the code and will be initialized in if __name__ == '__main__':
vocab = None
VOCAB_SIZE = None
PAD_IDX = None
UNK_IDX = None
NUM_CLASSES = None # Will be set based on dataset

# --- Logging Setup ---
# Clear existing handlers if any (useful for re-running in notebooks)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_filename = f'qsa_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# --- Global Configurations (Adjust as needed for your paper) ---
logger.info("Initializing Quantum Self-Attention Model")
logger.info("Loading configurations and setting up environment...")

# Model Hyperparameters
EMBED_DIM = 2           # Embedding dimension - MUST BE A POWER OF 2 for Amplitude Encoding
MAX_LEN = 20            # Maximum sequence length - Adjust based on chosen dataset
                        # AA Softmax is currently fixed to this MAX_LEN.
# Training Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 0.0005  # Reduced learning rate
NUM_EPOCHS = 5
GRAD_CLIP_VALUE = 1.0   # For gradient clipping

# Quantum Circuit Hyperparameters
N_SHOTS_QIP = 100
N_SHOTS_AA = 100
AA_ITERATIONS = 1

# Dataset specific
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# --- File path for the dataset ---
# !!! USER ACTION: Download SMSSpamCollection and place it in the same directory or provide the correct path !!!
SMS_DATASET_PATH = "SMSSpamCollection.tsv" 

# --- Quantum Setup ---
if not (EMBED_DIM > 0 and (EMBED_DIM & (EMBED_DIM - 1) == 0)):
    raise ValueError("EMBED_DIM must be a power of 2 for simplified Amplitude Encoding.")

N_QUBITS_AE_ENCODING = int(math.log2(EMBED_DIM))
N_QUBITS_ANCILLA_QIP = 1
N_QUBITS_AA_SOFTMAX = (MAX_LEN - 1).bit_length() if MAX_LEN > 0 else 0

dev_qip = qml.device("default.qubit", 
                     wires=(N_QUBITS_ANCILLA_QIP + N_QUBITS_AE_ENCODING), 
                     shots=N_SHOTS_QIP)
dev_aa_softmax = qml.device("default.qubit", wires=N_QUBITS_AA_SOFTMAX, shots=N_SHOTS_AA)

# --- Custom Vocabulary and Tokenizer ---
class SimpleTokenizer:
    def __call__(self, text_string):
        text_string = re.sub(r"[^a-zA-Z0-9\s]", "", text_string) # Remove special chars
        return text_string.lower().split()

class Vocab:
    def __init__(self, counter, specials, min_freq=1):
        self.specials = specials
        self._stoi = {token: i for i, token in enumerate(specials)}
        self._itos = list(specials)
        
        idx_offset = len(specials)
        for token, freq in counter.items():
            if freq >= min_freq:
                self._stoi[token] = len(self._itos) 
                self._itos.append(token)
        
        self.default_index = self._stoi.get(UNK_TOKEN, 0) 

    def __call__(self, tokens): 
        return [self._stoi.get(token, self.default_index) for token in tokens]

    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        # Ensure itos is consistent with stoi if built dynamically
        if len(self._itos) != len(self._stoi):
             current_len = len(self._itos)
             needed_len = len(self._stoi)
             if needed_len > current_len:
                 self._itos.extend([None] * (needed_len - current_len))
             elif needed_len < current_len: 
                 self._itos = self._itos[:needed_len]
             for s, i in self._stoi.items():
                 if i < len(self._itos): 
                    self._itos[i] = s
        return self._itos
        
    def __len__(self):
        return len(self._itos)

    def set_default_index(self, index): 
        self.default_index = index

tokenizer = SimpleTokenizer()
# vocab will be initialized in if __name__ == '__main__':

# --- Custom Dataset Class for CSV/TSV data ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab_obj, tokenizer_obj, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab_obj
        self.tokenizer = tokenizer_obj
        self.max_len = max_len
        self.pad_idx = self.vocab.get_stoi()[PAD_TOKEN]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenizer(text)
        indexed_text = self.vocab(tokenized_text)
        if len(indexed_text) > self.max_len:
            indexed_text = indexed_text[:self.max_len]
        else:
            indexed_text.extend([self.pad_idx] * (self.max_len - len(indexed_text)))
        return torch.tensor(indexed_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_batch_custom(batch):
    label_list, text_list = [], []
    for (text_tensor, label_tensor) in batch: 
        label_list.append(label_tensor)
        text_list.append(text_tensor) 
    return torch.stack(text_list), torch.tensor(label_list, dtype=torch.long)


# --- Quantum Components (Pennylane) ---
@qml.qnode(dev_qip, interface="torch", diff_method="parameter-shift")
def hadamard_test_re_inner_product_circuit(q_features, k_features):
    ancilla_wire = 0
    encoding_wires = list(range(N_QUBITS_ANCILLA_QIP, N_QUBITS_ANCILLA_QIP + N_QUBITS_AE_ENCODING))
    qml.Hadamard(wires=ancilla_wire)
    qml.PauliX(wires=ancilla_wire)
    qml.ctrl(lambda features: qml.AmplitudeEmbedding(features=features, wires=encoding_wires, normalize=True, pad_with=0.), 
             control=ancilla_wire)(features=q_features)
    qml.PauliX(wires=ancilla_wire)
    qml.ctrl(lambda features: qml.AmplitudeEmbedding(features=features, wires=encoding_wires, normalize=True, pad_with=0.), 
             control=ancilla_wire)(features=k_features)
    qml.Hadamard(wires=ancilla_wire)
    return qml.probs(wires=ancilla_wire)

@qml.qnode(dev_aa_softmax, interface="torch", diff_method="parameter-shift")
def partial_amplitude_amplification_softmax(initial_probabilities):
    num_input_states = len(initial_probabilities)
    if num_input_states != MAX_LEN:
        raise ValueError(f"partial_amplitude_amplification_softmax expects {MAX_LEN} initial probabilities, got {num_input_states}.")
    target_wires = list(range(N_QUBITS_AA_SOFTMAX))
    expected_dim_for_stateprep = 2**N_QUBITS_AA_SOFTMAX
    
    initial_probs_numpy = initial_probabilities.detach().cpu().numpy()
    initial_probs_clipped = pnp.clip(initial_probs_numpy, 0, 1) 
    
    sqrt_probs_cpu = pnp.sqrt(initial_probs_clipped) 
    
    if num_input_states < expected_dim_for_stateprep:
        padding_len = expected_dim_for_stateprep - num_input_states
        sqrt_probs_padded = pnp.pad(sqrt_probs_cpu, (0, padding_len), mode='constant', constant_values=0.0)
    else:
        sqrt_probs_padded = sqrt_probs_cpu
        
    norm_sqrt_probs = pnp.linalg.norm(sqrt_probs_padded)
    if norm_sqrt_probs < 1e-9:
        # logger.warning("sqrt_probs_padded has zero norm in AA softmax. Using uniform state.")
        sqrt_probs_padded = pnp.ones(expected_dim_for_stateprep) / pnp.sqrt(expected_dim_for_stateprep)

    qml.StatePrep(sqrt_probs_padded, wires=target_wires)
    
    mean_prob_val = pnp.mean(initial_probs_clipped) 
    diag_phases = pnp.ones(expected_dim_for_stateprep) 
    for j_idx in range(num_input_states):
        if initial_probs_clipped[j_idx] > mean_prob_val: 
            diag_phases[j_idx] = -1.0
    qml.DiagonalQubitUnitary(diag_phases, wires=target_wires)
    
    for _ in range(AA_ITERATIONS): 
        qml.adjoint(qml.StatePrep(sqrt_probs_padded, wires=target_wires))
        reflection_0_phases = pnp.ones(expected_dim_for_stateprep)
        reflection_0_phases[0] = -1.0
        qml.DiagonalQubitUnitary(reflection_0_phases, wires=target_wires)
        qml.StatePrep(sqrt_probs_padded, wires=target_wires)
    return qml.probs(wires=target_wires)

# --- Hybrid Model (PyTorch + Pennylane) ---
class QuantumSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_shots_qip, use_quantum_softmax=False, aa_iterations=1):
        super().__init__()
        if not (embed_dim > 0 and (embed_dim & (embed_dim - 1) == 0)):
            raise ValueError("embed_dim must be a power of 2 for Amplitude Encoding.")
        self.embed_dim = embed_dim
        self.d_k = embed_dim 
        self.n_qubits_ae_encoding = int(math.log2(embed_dim))
        self.n_shots_qip = n_shots_qip
        self.sqrt_d_k = math.sqrt(self.d_k)
        self.use_quantum_softmax = use_quantum_softmax
        self.aa_iterations = aa_iterations 

        self.W_q = nn.Linear(embed_dim, self.d_k, bias=False)
        self.W_k = nn.Linear(embed_dim, self.d_k, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q.weight, gain=1.0) 
        nn.init.xavier_uniform_(self.W_k.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_v.weight, gain=1.0)

    def _prepare_features_for_amplitude_encoding(self, params_tensor):
        current_device = params_tensor.device
        current_dtype = params_tensor.dtype

        if not torch.isfinite(params_tensor).all():
            # logger.warning(f"Non-finite (NaN/inf) detected in _prepare_features. Input: {params_tensor.detach().cpu().numpy()}. Using default random vector.")
            default_vec = torch.randn(self.embed_dim, device=current_device, dtype=current_dtype) * 0.01
            if torch.linalg.norm(default_vec) < 1e-9 and self.embed_dim > 0:
                default_vec[0] = 0.1 
            return default_vec.clone() 

        norm = torch.linalg.norm(params_tensor)
        if norm < 1e-9: 
            # logger.warning(f"Near-zero vector detected in _prepare_features (norm: {norm}). Input: {params_tensor.detach().cpu().numpy()}. Using default random vector.")
            default_vec = torch.randn(self.embed_dim, device=current_device, dtype=current_dtype) * 0.01
            if torch.linalg.norm(default_vec) < 1e-9 and self.embed_dim > 0:
                default_vec[0] = 0.1
            return default_vec.clone() 
        
        return params_tensor.clone() 

    def _relu_normalize_l1(self, scores_1d):
        if not isinstance(scores_1d, torch.Tensor):
            current_device = self.W_q.weight.device if hasattr(self.W_q, 'weight') and self.W_q.weight is not None else 'cpu'
            scores_1d = torch.tensor(scores_1d, device=current_device, dtype=torch.float32)
        
        relu_scores = torch.relu(scores_1d)
        l1_norm = torch.sum(relu_scores)
        if l1_norm.item() < 1e-9: 
            num_elements = scores_1d.shape[0]
            if num_elements == 0: 
                 return torch.tensor([], device=scores_1d.device, dtype=torch.float32)
            return torch.ones_like(relu_scores, device=scores_1d.device) / num_elements
        else:
            return relu_scores / l1_norm 

    def forward(self, x_embed):
        batch_size, seq_len, embed_dim_runtime = x_embed.shape
        
        if embed_dim_runtime != self.embed_dim:
            raise ValueError(f"Input embedding dimension ({embed_dim_runtime}) "
                             f"does not match model's ({self.embed_dim})")
        
        if self.use_quantum_softmax and seq_len != MAX_LEN:
             raise ValueError(f"Quantum softmax is strictly configured for MAX_LEN={MAX_LEN}, "
                              f"but got input seq_len={seq_len}. Pad or truncate input sequences to MAX_LEN.")

        q_classical = self.W_q(x_embed) 
        k_classical = self.W_k(x_embed) 
        v_classical = self.W_v(x_embed) 

        if torch.isnan(q_classical).any() or torch.isinf(q_classical).any() or \
           torch.isnan(k_classical).any() or torch.isinf(k_classical).any() or \
           torch.isnan(v_classical).any() or torch.isinf(v_classical).any():
            logger.error("NaN/inf detected in Q, K, or V classical projections. Returning NaN tensors.")
            nan_result_val = torch.full_like(v_classical, float('nan'))
            nan_result_attn = torch.full((batch_size, seq_len, seq_len), float('nan'), device=x_embed.device)
            return nan_result_val, nan_result_attn


        attention_scores_or_weights_tensor = torch.zeros(batch_size, seq_len, seq_len, device=x_embed.device)

        for b_idx in range(batch_size):
            q_seq_single = q_classical[b_idx] 
            k_seq_single = k_classical[b_idx] 
            
            for i in range(seq_len):
                q_i_features_raw = q_seq_single[i]
                q_i_features = self._prepare_features_for_amplitude_encoding(q_i_features_raw)
                current_q_raw_scores = torch.zeros(seq_len, device=x_embed.device)
                for j in range(seq_len):
                    k_j_features_raw = k_seq_single[j]
                    k_j_features = self._prepare_features_for_amplitude_encoding(k_j_features_raw)
                    
                    try:
                        probs_ancilla = hadamard_test_re_inner_product_circuit(q_i_features, k_j_features)
                    except Exception as e:
                        logger.error(f"ERROR in QNode hadamard_test_re_inner_product_circuit at b{b_idx},i{i},j{j}: {e}")
                        logger.error(f"  Q_features (input to QNode): {q_i_features.detach().cpu().numpy()}")
                        logger.error(f"  K_features (input to QNode): {k_j_features.detach().cpu().numpy()}")
                        probs_ancilla = torch.tensor([0.5, 0.5], device=x_embed.device, dtype=q_i_features.dtype) 

                    z_ij = probs_ancilla[0] - probs_ancilla[1]
                    if torch.isnan(z_ij).any() or torch.isinf(z_ij).any(): 
                        z_ij = torch.tensor(0.0, device=z_ij.device, dtype=q_i_features.dtype) 
                    current_q_raw_scores[j] = z_ij
                
                if self.use_quantum_softmax:
                    initial_probs_for_aa = self._relu_normalize_l1(current_q_raw_scores) 
                    if torch.isnan(initial_probs_for_aa).any() or torch.isinf(initial_probs_for_aa).any():
                        initial_probs_for_aa = torch.ones_like(initial_probs_for_aa) / (initial_probs_for_aa.shape[0] + 1e-9) 

                    try:
                        aa_output_probs_all_states = partial_amplitude_amplification_softmax(initial_probs_for_aa)
                    except Exception as e:
                        logger.error(f"ERROR in QNode partial_amplitude_amplification_softmax at b{b_idx},i{i}: {e}")
                        logger.error(f"  Initial_probs_for_aa: {initial_probs_for_aa.detach().cpu().numpy()}")
                        aa_output_probs_all_states = torch.ones(2**N_QUBITS_AA_SOFTMAX, device=x_embed.device, dtype=initial_probs_for_aa.dtype) / (2**N_QUBITS_AA_SOFTMAX)

                    aa_weights = aa_output_probs_all_states[:seq_len] 
                    final_weights_sum = torch.sum(aa_weights)
                    if final_weights_sum.item() < 1e-9 : 
                        final_weights = torch.ones_like(aa_weights) / (seq_len + 1e-9) 
                    else:
                        final_weights = aa_weights / (final_weights_sum + 1e-9)

                    if torch.isnan(final_weights).any() or torch.isinf(final_weights).any():
                        final_weights = torch.ones_like(final_weights) / (seq_len + 1e-9) 
                    attention_scores_or_weights_tensor[b_idx, i, :] = final_weights
                else:
                    attention_scores_or_weights_tensor[b_idx, i, :] = current_q_raw_scores
        
        if self.use_quantum_softmax:
            attention_weights = attention_scores_or_weights_tensor 
        else:
            if torch.isnan(attention_scores_or_weights_tensor).any() or torch.isinf(attention_scores_or_weights_tensor).any():
                attention_scores_or_weights_tensor = torch.where(torch.isfinite(attention_scores_or_weights_tensor), 
                                                                 attention_scores_or_weights_tensor,
                                                                 torch.zeros_like(attention_scores_or_weights_tensor))
            attention_weights = nn.functional.softmax(attention_scores_or_weights_tensor / (self.sqrt_d_k + 1e-9), dim=-1) 
        
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            attention_weights = torch.ones_like(attention_weights) / (attention_weights.shape[-1] + 1e-9) 

        context_vectors = torch.bmm(attention_weights, v_classical)
        if torch.isnan(context_vectors).any() or torch.isinf(context_vectors).any():
            logger.error("NaN/inf detected in context_vectors. Propagating NaN.")
            return torch.full_like(v_classical, float('nan')), attention_weights

        return context_vectors, attention_weights

class QSATextClassifier(nn.Module):
    def __init__(self, vocab_size_param, embed_dim_param, num_classes_param, n_shots_qip_param, 
                 use_quantum_softmax_param=False, aa_iterations_param=1, pad_idx_param=0): 
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size_param, embed_dim_param, padding_idx=pad_idx_param)
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02) 
        with torch.no_grad(): 
            if pad_idx_param < self.embedding_layer.weight.size(0): 
                 self.embedding_layer.weight[pad_idx_param].zero_()

        self.qsa_layer = QuantumSelfAttentionLayer(embed_dim_param, n_shots_qip_param, 
                                                   use_quantum_softmax_param, aa_iterations_param)
        self.fc_out = nn.Linear(embed_dim_param, num_classes_param)
        self._init_fc_weights() 

    def _init_fc_weights(self): 
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0) 
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, text_batch_ids): 
        embedded_text = self.embedding_layer(text_batch_ids) 
        
        if torch.isnan(embedded_text).any() or torch.isinf(embedded_text).any():
            logger.error("NaN/inf detected in embedded_text after embedding layer. Propagating NaN.")
            nan_logits = torch.full((text_batch_ids.shape[0], self.fc_out.out_features), float('nan'), device=text_batch_ids.device)
            nan_attn = torch.full((text_batch_ids.shape[0], MAX_LEN, MAX_LEN), float('nan'), device=text_batch_ids.device)
            return nan_logits, nan_attn
        
        # Normalize each embedding vector in the batch
        # Calculate norm for each vector in the batch along the embedding dimension
        norm_embedded = torch.linalg.norm(embedded_text, ord=2, dim=-1, keepdim=True) # Corrected: ord=2
        # Add epsilon to prevent division by zero for zero-norm vectors (e.g. all padding)
        embedded_text_normalized = embedded_text / (norm_embedded + 1e-9) 

        # Check again after normalization
        if torch.isnan(embedded_text_normalized).any() or torch.isinf(embedded_text_normalized).any():
            # This can happen if a row in embedded_text was all zeros, leading to norm_embedded being zero for that row.
            # Replace NaN/inf with zeros or small random numbers.
            # logger.warning("NaN/inf detected in embedded_text after normalization. Replacing with zeros.")
            mask = ~torch.isfinite(embedded_text_normalized)
            embedded_text_normalized = torch.where(mask, torch.zeros_like(embedded_text_normalized), embedded_text_normalized)
            # If still NaN (e.g. if original was already NaN), propagate
            if torch.isnan(embedded_text_normalized).any() or torch.isinf(embedded_text_normalized).any():
                 logger.error("Persistent NaN/inf in embedded_text_normalized. Propagating NaN.")
                 nan_logits = torch.full((text_batch_ids.shape[0], self.fc_out.out_features), float('nan'), device=text_batch_ids.device)
                 nan_attn = torch.full((text_batch_ids.shape[0], MAX_LEN, MAX_LEN), float('nan'), device=text_batch_ids.device)
                 return nan_logits, nan_attn


        context_vectors, attn_weights = self.qsa_layer(embedded_text_normalized)
        
        if torch.isnan(context_vectors).any() or torch.isinf(context_vectors).any():
            logger.error("NaN/inf detected in context_vectors from QSA layer. Propagating NaN.")
            nan_logits = torch.full((text_batch_ids.shape[0], self.fc_out.out_features), float('nan'), device=text_batch_ids.device)
            return nan_logits, attn_weights 

        pooled_context = torch.mean(context_vectors, dim=1) 
        output_logits = self.fc_out(pooled_context) 
        return output_logits, attn_weights


# --- Training and Evaluation Function ---
def train_and_evaluate_model(model_config_name, use_quantum_softmax_flag, train_loader, test_loader,
                             current_vocab_size, current_num_classes, current_pad_idx): 
    logger.info(f"\n{'='*20} Starting Model Training {'='*20}")
    logger.info(f"Model Configuration:")
    logger.info(f"  - Model Name: {model_config_name}")
    logger.info(f"  - Quantum Softmax: {use_quantum_softmax_flag}")
    # ... (other logging as before)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = QSATextClassifier(current_vocab_size, EMBED_DIM, current_num_classes, N_SHOTS_QIP, 
                              use_quantum_softmax_param=use_quantum_softmax_flag, 
                              aa_iterations_param=AA_ITERATIONS, 
                              pad_idx_param=current_pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")


    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 
               'val_precision': [], 'val_recall': [], 'val_f1': [],
               'epoch_train_time': [], 'epoch_eval_time': []}
    
    nan_detected_overall = False
    total_training_start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0
        num_batches = len(train_loader)
        
        if nan_detected_overall and epoch > 0: 
            logger.warning(f"Stopping training for {model_config_name} due to persistent NaN in previous epoch(s).")
            for _i_epoch_fill in range(epoch, NUM_EPOCHS): 
                history['train_loss'].append(float('nan')); history['epoch_train_time'].append(0)
                history['val_loss'].append(float('nan')); history['val_accuracy'].append(float('nan'))
                history['val_precision'].append(float('nan')); history['val_recall'].append(float('nan'))
                history['val_f1'].append(float('nan')); history['epoch_eval_time'].append(0)
            break 

        nan_detected_in_current_epoch = False

        for i, (texts, labels) in enumerate(train_loader): 
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions, _ = model(texts) 
            
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                logger.error(f"NaN/inf detected in predictions at Epoch {epoch+1}, Batch {i+1}. Stopping epoch.")
                nan_detected_in_current_epoch = True; nan_detected_overall = True
                break 

            loss = criterion(predictions, labels)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.error(f"NaN/inf detected in loss at Epoch {epoch+1}, Batch {i+1}. Stopping epoch.")
                nan_detected_in_current_epoch = True; nan_detected_overall = True
                break 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE) 
            optimizer.step()
            
            epoch_train_loss += loss.item()
            if (i + 1) % max(1, num_batches // 4) == 0: 
                 logger.info(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{num_batches}], Train Loss: {loss.item():.4f}")
        
        if nan_detected_in_current_epoch:
            history['train_loss'].append(float('nan'))
            history['epoch_train_time'].append(time.time() - epoch_start_time)
            history['val_loss'].append(float('nan')); history['val_accuracy'].append(float('nan'))
            history['val_precision'].append(float('nan')); history['val_recall'].append(float('nan'))
            history['val_f1'].append(float('nan')); history['epoch_eval_time'].append(0)
            logger.warning(f"  Epoch {epoch+1} terminated early for {model_config_name} due to NaN/inf values.")
            continue 

        epoch_train_time = time.time() - epoch_start_time
        avg_epoch_train_loss = epoch_train_loss / num_batches if num_batches > 0 else float('nan')
        history['train_loss'].append(avg_epoch_train_loss)
        history['epoch_train_time'].append(epoch_train_time)
        logger.info(f"  End of Epoch {epoch+1} - Avg Training Loss: {avg_epoch_train_loss:.4f} - Train Time: {epoch_train_time:.2f}s")

        eval_start_time = time.time()
        model.eval()
        epoch_val_loss = 0
        all_preds, all_labels = [], []
        nan_in_eval = False
        with torch.no_grad():
            for texts, labels in test_loader: 
                texts, labels = texts.to(device), labels.to(device)
                predictions, attn_weights = model(texts)
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    logger.warning("NaN/inf detected in validation predictions. Evaluation results for this epoch will be NaN.")
                    nan_in_eval = True; break 
                
                loss = criterion(predictions, labels)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                     logger.warning("NaN/inf detected in validation loss. Evaluation results for this epoch will be NaN.")
                     nan_in_eval = True; break
                epoch_val_loss += loss.item()
                predicted_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        eval_end_time = time.time()
        epoch_eval_time = eval_end_time - eval_start_time
        
        if nan_in_eval or not all_labels : 
            avg_epoch_val_loss = float('nan')
            accuracy, precision, recall, f1 = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            avg_epoch_val_loss = epoch_val_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
            accuracy = accuracy_score(all_labels, all_preds) * 100
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

        history['val_loss'].append(avg_epoch_val_loss)
        history['val_accuracy'].append(accuracy)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['epoch_eval_time'].append(epoch_eval_time)

        logger.info(f"  Test Results Epoch {epoch+1}: Avg Val Loss: {avg_epoch_val_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f} - Eval Time: {epoch_eval_time:.2f}s")
    
    total_training_end_time = time.time()
    total_time = total_training_end_time - total_training_start_time
    logger.info(f"--- Finished: {model_config_name} --- Total Training & Eval Time: {total_time:.2f}s ---")
    return history, total_time

# --- Plotting and Analysis Functions ---
def plot_metrics(history, model_name):
    valid_epochs_count = sum(1 for loss_val in history['train_loss'] if not math.isnan(loss_val))
    if valid_epochs_count == 0:
        logger.warning(f"No valid data to plot for {model_name} (all epochs resulted in NaN).")
        return
    
    epochs_range = range(1, valid_epochs_count + 1) 

    def get_valid_metric(metric_list):
        # Ensure we only take up to valid_epochs_count and then filter NaNs
        return [m for m in metric_list[:valid_epochs_count] if not math.isnan(m)]

    train_loss_valid = get_valid_metric(history['train_loss'])
    # For validation metrics, their valid length might be less than train_loss_valid if NaNs occurred earlier in eval
    val_loss_valid = get_valid_metric(history['val_loss'])
    val_accuracy_valid = get_valid_metric(history['val_accuracy'])
    val_f1_valid = get_valid_metric(history['val_f1'])
    val_precision_valid = get_valid_metric(history['val_precision'])
    val_recall_valid = get_valid_metric(history['val_recall'])
    epoch_train_time_valid = get_valid_metric(history['epoch_train_time'])
    epoch_eval_time_valid = get_valid_metric(history['epoch_eval_time'])
    
    # Adjust plot ranges based on actual valid data points
    epochs_range_train_loss = range(1, len(train_loss_valid) + 1)
    epochs_range_val_loss = range(1, len(val_loss_valid) + 1)
    epochs_range_val_acc = range(1, len(val_accuracy_valid) + 1)
    epochs_range_val_f1 = range(1, len(val_f1_valid) + 1)
    epochs_range_val_prec = range(1, len(val_precision_valid) + 1)
    epochs_range_val_rec = range(1, len(val_recall_valid) + 1)
    epochs_range_time = range(1, len(epoch_train_time_valid) + 1)


    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    if train_loss_valid: plt.plot(epochs_range_train_loss, train_loss_valid, 'bo-', label='Training Loss')
    if val_loss_valid: plt.plot(epochs_range_val_loss, val_loss_valid, 'ro-', label='Validation Loss') 
    plt.title(f'Loss ({model_name})'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(2, 3, 2)
    if val_accuracy_valid: plt.plot(epochs_range_val_acc, val_accuracy_valid, 'go-', label='Validation Accuracy')
    plt.title(f'Accuracy ({model_name})'); plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend()

    plt.subplot(2, 3, 3)
    if val_f1_valid: plt.plot(epochs_range_val_f1, val_f1_valid, 'mo-', label='Validation F1-score (Weighted)')
    plt.title(f'F1-score ({model_name})'); plt.xlabel('Epochs'); plt.ylabel('F1-score'); plt.legend()
    
    plt.subplot(2, 3, 4)
    if val_precision_valid: plt.plot(epochs_range_val_prec, val_precision_valid, 'co-', label='Validation Precision (Weighted)')
    plt.title(f'Precision ({model_name})'); plt.xlabel('Epochs'); plt.ylabel('Precision'); plt.legend()
    
    plt.subplot(2, 3, 5)
    if val_recall_valid: plt.plot(epochs_range_val_rec, val_recall_valid, 'yo-', label='Validation Recall (Weighted)')
    plt.title(f'Recall ({model_name})'); plt.xlabel('Epochs'); plt.ylabel('Recall'); plt.legend()
    
    plt.subplot(2, 3, 6)
    if epoch_train_time_valid: plt.plot(epochs_range_time, epoch_train_time_valid, 'bs-', label='Epoch Train Time')
    if epoch_eval_time_valid and len(epoch_eval_time_valid) == len(epoch_train_time_valid): # Ensure same length for plotting
         plt.plot(epochs_range_time, epoch_eval_time_valid, 'rs-', label='Epoch Eval Time')
    plt.title(f'Time per Epoch ({model_name})'); plt.xlabel('Epochs'); plt.ylabel('Time (s)'); plt.legend()
    
    plt.tight_layout(); plt.savefig(f"metrics_{model_name.replace(' ', '_')}.png")
    logger.info(f"Metrics plot saved to metrics_{model_name.replace(' ', '_')}.png")

def visualize_attention(text_ids_sample, attention_weights, model_name, current_vocab, sample_idx=0):
    if attention_weights is None or len(attention_weights) == 0 or torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
        logger.warning(f"Cannot visualize attention for {model_name} due to invalid attention_weights.")
        return
    
    itos_map = current_vocab.get_itos()
    text_tokens = []
    if not isinstance(itos_map, list):
        try:
            max_idx = 0
            if itos_map: 
                max_idx = max(itos_map.keys()) if isinstance(itos_map, dict) else len(itos_map) -1
            temp_list = [UNK_TOKEN] * (max_idx + 1)
            for idx, token_str in (itos_map.items() if isinstance(itos_map, dict) else enumerate(itos_map)):
                temp_list[idx] = token_str
            itos_map = temp_list
        except Exception as e: 
            logger.warning(f"vocab.get_itos() returned an unexpected type or error: {e}. Cannot map token IDs to strings for visualization.")
            text_tokens = [str(token_idx.item()) for token_idx in text_ids_sample]
    
    if isinstance(itos_map, list): # Should be a list now
        text_tokens = [itos_map[token_idx.item()] if token_idx.item() < len(itos_map) else UNK_TOKEN for token_idx in text_ids_sample]
    elif not text_tokens: # If it wasn't converted and wasn't already populated
        text_tokens = [str(token_idx.item()) for token_idx in text_ids_sample]

    attn_map = attention_weights[sample_idx].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(min(MAX_LEN + 2, 10) , min(MAX_LEN + 2, 8)))
    im = ax.imshow(attn_map, cmap='viridis')
    ax.set_xticks(range(len(text_tokens))); ax.set_yticks(range(len(text_tokens)))
    ax.set_xticklabels(text_tokens, rotation=90, fontsize=8); ax.set_yticklabels(text_tokens, fontsize=8)
    ax.set_xlabel("Key Tokens"); ax.set_ylabel("Query Tokens")
    plt.colorbar(im); plt.title(f"Attention Map - {model_name} (Sample {sample_idx})"); plt.tight_layout()
    plt.savefig(f"attention_map_{model_name.replace(' ', '_')}_sample_{sample_idx}.png")
    logger.info(f"Attention map saved to attention_map_{model_name.replace(' ', '_')}_sample_{sample_idx}.png")

# --- Dummy Dataset Creation (if real dataset loading fails or is not used) ---
def create_dummy_dataset_for_run(num_samples, max_len, vocab_size_param, num_classes_param, pad_idx_param):
    texts_list = [torch.randint(0, vocab_size_param, (max_len,)) for _ in range(num_samples)]
    texts_tensor = torch.stack(texts_list)
    labels_tensor = torch.randint(0, num_classes_param, (num_samples,))
    return TensorDataset(texts_tensor, labels_tensor) 


# --- Main Execution ---
if __name__ == '__main__':
    logger.info(f"Starting Quantum Self-Attention Text Classification. Log file: {log_filename}")
    logger.info("Loading and preprocessing data...")
    
    try:
        logger.info(f"Attempting to load dataset from: {SMS_DATASET_PATH}")
        df = pd.read_csv(SMS_DATASET_PATH, sep='\t', header=None, names=['label', 'message'])
        logger.info(f"Successfully loaded dataset with {len(df)} samples")
        
        logger.info("Preprocessing data...")
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['message'] = df['message'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
        logger.info("Data preprocessing completed")

        all_texts = df['message'].tolist()
        all_labels = df['label'].tolist()

        logger.info("Building vocabulary...")
        token_counter = Counter()
        for text in all_texts:
            token_counter.update(tokenizer(text))
        logger.info(f"Found {len(token_counter)} unique tokens")

        config.vocab = Vocab(token_counter, specials=[PAD_TOKEN, UNK_TOKEN], min_freq=2)
        config.VOCAB_SIZE = len(config.vocab)
        config.PAD_IDX = config.vocab.get_stoi()[PAD_TOKEN]
        config.UNK_IDX = config.vocab.get_stoi()[UNK_TOKEN]
        config.NUM_CLASSES = 2 

        logger.info(f"Vocabulary built:")
        logger.info(f"  Size: {config.VOCAB_SIZE}")
        logger.info(f"  PAD_IDX: {config.PAD_IDX}")
        logger.info(f"  UNK_IDX: {config.UNK_IDX}")

        logger.info("Splitting dataset...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        logger.info(f"Dataset split complete:")
        logger.info(f"  Training samples: {len(train_texts)}")
        logger.info(f"  Testing samples: {len(test_texts)}")

        train_dataset_real = TextDataset(train_texts, train_labels, config.vocab, tokenizer, MAX_LEN)
        test_dataset_real = TextDataset(test_texts, test_labels, config.vocab, tokenizer, MAX_LEN)
        train_loader = DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_custom)
        test_loader = DataLoader(test_dataset_real, batch_size=BATCH_SIZE, collate_fn=collate_batch_custom)
        logger.info(f"Data loaders created with batch size: {BATCH_SIZE}")
        DATASET_TYPE = "SMS_Spam"

    except FileNotFoundError:
        logger.warning(f"Dataset file not found at {SMS_DATASET_PATH}")
        logger.info("Falling back to dummy dataset...")
        dummy_vocab_map = {f"word{i}": i for i in range(1000 - 2)} 
        config.vocab = Vocab(dummy_vocab_map, specials=[PAD_TOKEN, UNK_TOKEN])
        config.VOCAB_SIZE = len(config.vocab)
        config.PAD_IDX = config.vocab.get_stoi()[PAD_TOKEN]
        config.UNK_IDX = config.vocab.get_stoi()[UNK_TOKEN]
        config.NUM_CLASSES = 2 
        train_dataset_dummy = create_dummy_dataset_for_run(128, MAX_LEN, config.VOCAB_SIZE, config.NUM_CLASSES, config.PAD_IDX)
        test_dataset_dummy = create_dummy_dataset_for_run(64, MAX_LEN, config.VOCAB_SIZE, config.NUM_CLASSES, config.PAD_IDX)
        train_loader = DataLoader(train_dataset_dummy, batch_size=BATCH_SIZE, shuffle=True) 
        test_loader = DataLoader(test_dataset_dummy, batch_size=BATCH_SIZE)
        logger.info(f"Using Dummy Dataset. VOCAB_SIZE: {config.VOCAB_SIZE}, NUM_CLASSES: {config.NUM_CLASSES}")
        DATASET_TYPE = "Dummy"
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}", exc_info=True)
        logger.info("Falling back to dummy dataset...")
        dummy_vocab_map = {f"word{i}": i for i in range(1000 - 2)}
        config.vocab = Vocab(dummy_vocab_map, specials=[PAD_TOKEN, UNK_TOKEN])
        config.VOCAB_SIZE = len(config.vocab)
        config.PAD_IDX = config.vocab.get_stoi()[PAD_TOKEN]
        config.UNK_IDX = config.vocab.get_stoi()[UNK_TOKEN]
        config.NUM_CLASSES = 2
        train_dataset_dummy = create_dummy_dataset_for_run(128, MAX_LEN, config.VOCAB_SIZE, config.NUM_CLASSES, config.PAD_IDX)
        test_dataset_dummy = create_dummy_dataset_for_run(64, MAX_LEN, config.VOCAB_SIZE, config.NUM_CLASSES, config.PAD_IDX)
        train_loader = DataLoader(train_dataset_dummy, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset_dummy, batch_size=BATCH_SIZE)
        logger.info(f"Using Dummy Dataset due to error. VOCAB_SIZE: {config.VOCAB_SIZE}, NUM_CLASSES: {config.NUM_CLASSES}")
        DATASET_TYPE = "Dummy (Error Fallback)"

    logger.info(f"\nModel Configuration:")
    logger.info(f"  Embedding dimension: {EMBED_DIM}")
    logger.info(f"  Maximum sequence length: {MAX_LEN}")
    logger.info(f"  Number of qubits for AE encoding: {N_QUBITS_AE_ENCODING}")
    logger.info(f"  QIP Circuit qubits: {N_QUBITS_ANCILLA_QIP + N_QUBITS_AE_ENCODING}")
    logger.info(f"  AA Softmax Circuit qubits: {N_QUBITS_AA_SOFTMAX}")
    logger.info(f"  Shots per QIP: {N_SHOTS_QIP}")
    logger.info(f"  Shots per AA Softmax: {N_SHOTS_AA}")
    logger.info(f"  AA Iterations: {AA_ITERATIONS}")
    
    if not train_loader or not test_loader:
        logger.error("Data loaders are not initialized. Exiting.")
        exit()
    
    logger.info(f"\nData loader status:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Testing batches: {len(test_loader)}")

    logger.info("\nStarting model evaluations...")
    results_summary = {}

    history_classical, time_classical = train_and_evaluate_model(
        model_config_name=f"QSA_Classical_Softmax_{DATASET_TYPE.replace(' ','_')}",
        use_quantum_softmax_flag=False,
        train_loader=train_loader,
        test_loader=test_loader,
        current_vocab_size=config.VOCAB_SIZE, 
        current_num_classes=config.NUM_CLASSES,
        current_pad_idx=config.PAD_IDX
    )
    results_summary["Classical Softmax"] = {
        "accuracy": history_classical['val_accuracy'][-1] if history_classical['val_accuracy'] and not math.isnan(history_classical['val_accuracy'][-1]) else 0.0, 
        "f1": history_classical['val_f1'][-1] if history_classical['val_f1'] and not math.isnan(history_classical['val_f1'][-1]) else 0.0,
        "precision": history_classical['val_precision'][-1] if history_classical['val_precision'] and not math.isnan(history_classical['val_precision'][-1]) else 0.0,
        "recall": history_classical['val_recall'][-1] if history_classical['val_recall'] and not math.isnan(history_classical['val_recall'][-1]) else 0.0,
        "training_time": time_classical
    }
    if history_classical['train_loss'] and not any(math.isnan(x) for x in history_classical['train_loss']): 
        plot_metrics(history_classical, f"QSA_Classical_Softmax_{DATASET_TYPE.replace(' ','_')}")

    history_quantum_aa, time_quantum_aa = train_and_evaluate_model(
        model_config_name=f"QSA_AA_Quantum_Softmax_{DATASET_TYPE.replace(' ','_')}",
        use_quantum_softmax_flag=True,
        train_loader=train_loader,
        test_loader=test_loader,
        current_vocab_size=config.VOCAB_SIZE,
        current_num_classes=config.NUM_CLASSES,
        current_pad_idx=config.PAD_IDX
    )
    results_summary["AA-based Quantum Softmax"] = {
        "accuracy": history_quantum_aa['val_accuracy'][-1] if history_quantum_aa['val_accuracy'] and not math.isnan(history_quantum_aa['val_accuracy'][-1]) else 0.0,
        "f1": history_quantum_aa['val_f1'][-1] if history_quantum_aa['val_f1'] and not math.isnan(history_quantum_aa['val_f1'][-1]) else 0.0,
        "precision": history_quantum_aa['val_precision'][-1] if history_quantum_aa['val_precision'] and not math.isnan(history_quantum_aa['val_precision'][-1]) else 0.0,
        "recall": history_quantum_aa['val_recall'][-1] if history_quantum_aa['val_recall'] and not math.isnan(history_quantum_aa['val_recall'][-1]) else 0.0,
        "training_time": time_quantum_aa
    }
    if history_quantum_aa['train_loss'] and not any(math.isnan(x) for x in history_quantum_aa['train_loss']):
        plot_metrics(history_quantum_aa, f"QSA_AA_Quantum_Softmax_{DATASET_TYPE.replace(' ','_')}")

    logger.info("\n--- Final Comparison Summary ---")
    for model_name, metrics in results_summary.items():
        logger.info(f"Model: {model_name} on {DATASET_TYPE} Dataset")
        logger.info(f"  Final Test Accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"  Final Test F1-score: {metrics['f1']:.4f}")
        logger.info(f"  Final Test Precision: {metrics['precision']:.4f}")
        logger.info(f"  Final Test Recall: {metrics['recall']:.4f}")
        logger.info(f"  Total Training & Eval Time: {metrics['training_time']:.2f}s")
        logger.info("-" * 30)

    try:
        if test_loader and len(test_loader) > 0:
            sample_texts, sample_labels = next(iter(test_loader)) 
            if sample_texts is not None and len(sample_texts) > 0:
                # For visualization, create fresh model instances (or load saved ones)
                device_viz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_classical_viz = QSATextClassifier(config.VOCAB_SIZE, EMBED_DIM, config.NUM_CLASSES, N_SHOTS_QIP, 
                                                    use_quantum_softmax_param=False, pad_idx_param=config.PAD_IDX).to(device_viz)
                model_classical_viz.eval() 
                with torch.no_grad():
                    _, attn_classical = model_classical_viz(sample_texts.to(device_viz)) 
                visualize_attention(sample_texts[0].cpu(), attn_classical.cpu(), f"QSA_Classical_Softmax_{DATASET_TYPE.replace(' ','_')}", config.vocab, sample_idx=0)

                model_quantum_aa_viz = QSATextClassifier(config.VOCAB_SIZE, EMBED_DIM, config.NUM_CLASSES, N_SHOTS_QIP, 
                                                     use_quantum_softmax_param=True, pad_idx_param=config.PAD_IDX).to(device_viz)
                model_quantum_aa_viz.eval()
                with torch.no_grad():
                    _, attn_quantum_aa = model_quantum_aa_viz(sample_texts.to(device_viz))
                visualize_attention(sample_texts[0].cpu(), attn_quantum_aa.cpu(), f"QSA_AA_Quantum_Softmax_{DATASET_TYPE.replace(' ','_')}", config.vocab, sample_idx=0)
            else:
                logger.warning("Test loader did not yield a valid batch for attention visualization.")
        else:
            logger.warning("Test loader is empty or not initialized, cannot visualize attention.")
    except StopIteration:
        logger.warning("Test loader was empty, cannot visualize attention.")
    except Exception as e:
        logger.error(f"Error during attention visualization: {e}", exc_info=True)

