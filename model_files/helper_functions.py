import torch
import torch.nn.functional as F
from collections import Counter
from nltk import word_tokenize
from typing import Dict
import os,sys
import numpy as np
from scipy.stats import spearmanr

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path+"/")#add to load modules
# __all__ = ["apply_whitening_batch","overlap_coeff_triplet","jaccard_overlap_triplet","containment_triplet"]
import numpy as np
from numpy.linalg import eigh
@torch.no_grad()
def matrix_entropy_from_gram(
    K: torch.Tensor,
    alpha: float = 1.0,                 # Rényi order; alpha=1 uses the Shannon/vN limit
    eps: float = 1e-12,                 # numerical stability
    eig_dtype: torch.dtype = torch.float32,  # promote for eig (avoid bf16/half issues)
    normalize_by_log_n: bool = True,    # divide by log(n_eff) so result ∈ [0,1]
    zero_eig_threshold: float = 1e-9    # threshold to count “nonzero” eigenvalues
) -> torch.Tensor:
    """
    Matrix-based Rényi entropy S_alpha for a *single* Gram matrix K = ZZ^T (Eq. 1),
    with optional normalization by log(n_eff).

    Args:
        K: (n, n) symmetric PSD Gram matrix.
        alpha: Rényi order (>0). If alpha==1, uses the Shannon/von Neumann limit.
        eps: Small constant for numerical stability.
        eig_dtype: Dtype used for eigendecomposition (float32/64 recommended).
        normalize_by_log_n: If True, divide by log(n_eff) for length invariance.
        zero_eig_threshold: Relative threshold to count nonzero eigenvalues for n_eff.

    Returns:
        Scalar tensor with S_alpha (normalized if normalize_by_log_n=True).
    """
    # Promote to a stable dtype for linear algebra
    K = K.to(dtype=eig_dtype)

    # Symmetrize (guard tiny asymmetries) and add a tiny ridge to ensure PSD
    K = 0.5 * (K + K.transpose(-1, -2))
    n = K.shape[0]
    if n == 0:
        return torch.zeros((), device=K.device, dtype=eig_dtype)
    K = K + eps * torch.eye(n, device=K.device, dtype=K.dtype)

    # Eigenvalues (PSD → use eigvalsh), clamp tiny negatives to 0
    evals = torch.linalg.eigvalsh(K).clamp_min(0)

    # Normalize spectrum by trace → a probability vector over eigenvalues
    trK = evals.sum() + eps
    p = evals / trK  # sum(p)=1

    # S_alpha per Eq. (1)
    if alpha == 1.0:
        H = -(p * torch.log(p + eps)).sum()
    else:
        H = (1.0 / (1.0 - alpha)) * torch.log((p.pow(alpha)).sum() + eps)

    # Optional normalization by log(n_eff) so H ∈ [0,1] (comparability across lengths)
    if normalize_by_log_n:
        # count “effective” nonzero eigenvalues (relative to max eigenvalue)
        lam_max = evals.max().item() if evals.numel() > 0 else 0.0
        rel_thresh = zero_eig_threshold * max(lam_max, eps)
        n_eff = max(int((evals > rel_thresh).sum().item()), 1)
        H = H / (torch.log(torch.tensor(float(n_eff), device=K.device, dtype=eig_dtype)) + eps)

    return H

  


@torch.no_grad()
def matrix_entropy_from_tokens(
    Z: torch.Tensor,                 # (L, D) token embeddings for one prompt
    alpha: float = 1.0,
    eps: float = 1e-12,
    eig_dtype: torch.dtype = torch.float32,
    drop_zero_rows: bool = True,     # drop fully padded rows if they are zeros
) -> torch.Tensor:
    """
    Convenience wrapper: build K = Z Z^T, then compute S_alpha via Eq. (1).
    """
    Z = Z.to(dtype=eig_dtype)
    if drop_zero_rows:
        mask = (Z.abs().sum(dim=-1) > 0)
        if not torch.any(mask):
            return torch.tensor(0.0, device=Z.device, dtype=eig_dtype)
        Z = Z[mask]

    K = Z @ Z.transpose(-1, -2)  # Gram
    return matrix_entropy_from_gram(K, alpha=alpha, eps=eps, eig_dtype=eig_dtype)


@torch.no_grad()
def matrix_entropy_batch_from_tokens(
    Z_batch: torch.Tensor,           # (B, L, D)
    alpha: float = 1.0,
    eps: float = 1e-12,
    eig_dtype: torch.dtype = torch.float32,
    drop_zero_rows: bool = True,
    return_input_dtype: bool = False,
) -> torch.Tensor:
    """
    Batched version: one entropy per sequence in the batch, exactly following Eq. (1).
    """
    assert Z_batch.dim() == 3, "Z_batch must be (B, L, D)"
    B = Z_batch.size(0)
    out = torch.empty(B, device=Z_batch.device, dtype=eig_dtype)
    for b in range(B):
        out[b] = matrix_entropy_from_tokens(
            Z_batch[b], alpha=alpha, eps=eps, eig_dtype=eig_dtype, drop_zero_rows=drop_zero_rows
        )
    return out.to(Z_batch.dtype) if return_input_dtype else out

# def prompt_entropy_batch_infer_mask(
#     token_states: torch.Tensor,   # (B, L, D), padded rows are exactly zero
#     alpha: float = 1.0,
#     eps: float = 1e-12,
#     empty_return: float = 0.0,    # value to return if a sequence has 0 valid tokens
#     eig_dtype: torch.dtype = torch.float32,
#     return_input_dtype: bool = False 
# ) -> torch.Tensor:

#     assert token_states.dim() == 3, "token_states must be (B, L, D)"
#     B, L, D = token_states.shape
#     device = token_states.device

#     # infer valid-token mask: True = keep
#     valid_mask = (token_states.abs().sum(dim=-1) > 0)  # (B, L) bool
#     # print("valid_mask",valid_mask)
#     # accumulate in eig_dtype for stability
#     out = torch.empty(B, device=device, dtype=eig_dtype)

#     for b in range(B):
#         idx = valid_mask[b]                 # (L,)
#         n = int(idx.sum().item())

#         if n == 0:
#             out[b] = torch.tensor(empty_return, device=device, dtype=eig_dtype)
#             continue

#         # slice valid tokens and PROMOTE dtype for stable linear algebra
#         Z = token_states[b, idx, :].to(dtype=eig_dtype)   # (n, D)
        
#         # Gram matrix on valid tokens
#         K = Z @ Z.transpose(-1, -2)                       # (n, n)
#         K = K + eps * torch.eye(n, device=device, dtype=eig_dtype)

#         # eigvals for symmetric PSD
#         evals = torch.linalg.eigvalsh(K)                  # (n,)
#         evals = torch.clamp(evals, min=0.0)

#         # normalize to simplex
#         p = evals / (evals.sum() + eps)

#         # entropy
#         if alpha == 1.0:
#             out[b] = -(p * torch.log(p + eps)).sum()
#         else:
#             out[b] = (1.0 / (1.0 - alpha)) * torch.log((p.pow(alpha)).sum() + eps)

#     return out.to(token_states.dtype) if return_input_dtype else out


# def prompt_entropy(token_states: torch.Tensor, alpha: float = 1.0, eps: float = 1e-12,eig_dtype: torch.dtype = torch.float32) -> torch.Tensor:
#     # Step 1. Compute Gram matrix K = Z Z^T (L x L)
#     Z = token_states.to(dtype=eig_dtype)

#     # Remove all-zero padded rows (if any)
#     valid_mask = (Z.abs().sum(dim=-1) > 0)
#     print("valid_mask",valid_mask)
#     if valid_mask.sum() == 0:
#         return torch.tensor(0.0, device=Z.device, dtype=eig_dtype)
#     Z = Z[valid_mask]

#     # Step 1. Gram matrix
#     K = Z @ Z.T

#     # Step 2. Add small ridge to diagonal for numerical stability
#     K = K + eps * torch.eye(K.size(0), device=K.device, dtype=K.dtype)

#     # Step 3. Eigenvalues (Hermitian, PSD)
#     evals = torch.linalg.eigvalsh(K)
#     evals = torch.clamp(evals, min=0.0)

#     # Step 4. Normalize eigenvalues
#     p = evals / (torch.sum(evals) + eps)

#     # Step 5. Entropy
#     if alpha == 1.0:
#         entropy = -(p * torch.log(p + eps)).sum()
#     else:
#         entropy = (1.0 / (1.0 - alpha)) * torch.log((p.pow(alpha)).sum() + eps)

#     return entropy


try:
    import spacy
    print("spacy loaded")
    _NLP = spacy.load("/home/hrk21/projects/def-hsajjad/hrk21/spacy/en_core_web_sm/en_core_web_sm-3.8.0")
except Exception:
    _NLP = None
    print("spacy failed")

# Fallback to NLTK if spaCy not available
if _NLP is None:
    import nltk
    # Make sure these are downloaded in your environment once:
    # nltk.download("punkt"); nltk.download("averaged_perceptron_tagger")
    # nltk.download("maxent_ne_chunker"); nltk.download("words")
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.tree import Tree

def _tokens_no_entities(text: str) -> list[str]:
    """
    Return lowercased tokens that are NOT part of any named entity.
    - spaCy path: uses token.ent_iob_ to filter
    - NLTK path: uses ne_chunk to remove NE subtrees
    """
    if not text:
        return []

    if _NLP is not None:
        doc = _NLP(text)
        # keep only tokens that are not inside a named entity
        return [t.text.lower() for t in doc if t.ent_iob_ == "O" and t.is_alpha]

    # --- NLTK fallback ---
    toks = word_tokenize(text)
    tagged = pos_tag(toks)
    chunked = ne_chunk(tagged, binary=False)  # Tree of (NE subtrees) and (word, POS) leaves

    non_entity_tokens = []
    for node in chunked:
        if isinstance(node, Tree):
            # This subtree is a named entity -> skip all its leaves
            continue
        # node is a (word, POS) tuple
        w = node[0]
        if w.isalpha():
            non_entity_tokens.append(w.lower())
    return non_entity_tokens


# ---------- helpers ----------
def _counts(s: str,type="no_entities") -> Counter:
    if(type=="no_entities"):
        return Counter(_tokens_no_entities(s))
    else:
        return Counter(word_tokenize(s))

def _overlap_coeff_pct(A: Counter, B: Counter) -> float:
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = min(len(A), len(B))
    return 100.0 * inter / denom

def _jaccard_overlap_pct(A: Counter, B: Counter) -> float:
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return 100.0 * inter / union if union else 0.0

def _containment_pct(A: Counter, B: Counter) -> float:
    # asymmetric “recall” on the anchor (A)
    if not A:
        return 0.0
    inter = len(A & B)
    return 100.0 * inter / len(A)

def _triplet_report(metric_fn, A: Counter, P: Counter, D: Counter) -> Dict[str, float | str]:
    ap = metric_fn(A, P)  # anchor–paraphrase
    ad = metric_fn(A, D)  # anchor–distractor
    pd = metric_fn(P, D)  # paraphrase–distractor

    # symmetric distractor closeness: whichever is closer to D (A or P)
    d_close = max(ad, pd)#maximum overlap
    closest = "anchor" if ad >= pd else "paraphrase"

    return {
        "anchor_paraphrase": ap,
        "anchor_distractor": ad,
        "paraphrase_distractor": pd,
        "distractor_closest_to": closest,  # for debugging/analysis
        "triplet_score": d_close -ap         # large when D is close to A or P, while A–P are dissimilar
    }

# ---------- triplet metrics ----------
def overlap_coeff_triplet(anchor: str, paraphrase: str, distractor: str, tokenize_type="no_entities") -> Dict[str, float]:
    A, P, D = _counts(anchor, tokenize_type), _counts(paraphrase, tokenize_type), _counts(distractor, tokenize_type)
    return _triplet_report(_overlap_coeff_pct, A, P, D)

def jaccard_overlap_triplet(anchor: str, paraphrase: str, distractor: str, tokenize_type="no_entities") -> Dict[str, float]:
    A, P, D = _counts(anchor, tokenize_type), _counts(paraphrase, tokenize_type), _counts(distractor, tokenize_type)
    return _triplet_report(_jaccard_overlap_pct, A, P, D)

def containment_triplet(anchor: str, paraphrase: str, distractor: str, tokenize_type="no_entities") -> Dict[str, float]:
    A, P, D = _counts(anchor, tokenize_type), _counts(paraphrase, tokenize_type), _counts(distractor, tokenize_type)
    return _triplet_report(_containment_pct, A, P, D)

def run_all(anchor: str, paraphrase: str, distractor: str, tokenize_type="no_entities") -> Dict[str, float]:
    A, P, D = _counts(anchor, tokenize_type), _counts(paraphrase, tokenize_type), _counts(distractor, tokenize_type)

    return {
        "overlap_coeff":  _triplet_report(_overlap_coeff_pct, A, P, D)["triplet_score"],
        "jaccard": _triplet_report(_jaccard_overlap_pct, A, P, D)["triplet_score"],
        "containment": _triplet_report(_containment_pct, A, P, D)["triplet_score"],
    }

@torch.no_grad()
def apply_whitening_batch(
    embeddings: torch.Tensor,     # shape [batch_size, d]
    stats: dict,                  # dict from FullWhiteningStats.finalize()
    variant: str = "pca",         # "pca" or "zca"
    l2_after: bool = True,        # normalize after whitening (common for retrieval)
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Returns a batch of whitened embeddings (shape [batch_size, d]).
    NOTE: Pass the *raw* (unnormalized) embeddings if your stats were computed on raw embeddings.
    """
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be shape [batch_size, d].")

    device = embeddings.device if device is None else torch.device(device)

    mean_vec = stats["mean"].to(device, embeddings.dtype)  # [d]

    # Pick the whitener
    if variant == "pca":
        W = stats["pca_whitener"].to(device, embeddings.dtype)   # [d, d]
    elif variant == "zca":
        W = stats["zca_whitener"].to(device, embeddings.dtype)   # [d, d]
    else:
        raise ValueError("variant must be 'pca' or 'zca'")

    # Apply whitening: subtract mean, then project
    y = (embeddings - mean_vec) @ W    # [batch_size, d]

    # Optional L2 normalize
    if l2_after:
        y = F.normalize(y, p=2, dim=1)

    return y   # [batch_size, d]




def los_q1_q4_gap(jaccard: np.ndarray, margins: np.ndarray, q: float = 0.25) -> float:
    """LOS_sensitivity = mean(m | top quartile) - mean(m | bottom quartile)."""
    q1 = np.quantile(jaccard, q)
    q4 = np.quantile(jaccard, 1 - q)
    low  = margins[jaccard <= q1]
    high = margins[jaccard >= q4]
    if low.size == 0 or high.size == 0:
        return np.nan
    return float(high.mean() - low.mean())

def bootstrap_ci_spearman(
    jaccard: np.ndarray,
    margins: np.ndarray,
    B: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
):
    """Percentile CI for Spearman rho."""
    jaccard = np.asarray(jaccard, float)
    margins = np.asarray(margins, float)
    # point estimate
    rho_hat, _ = spearmanr(jaccard, margins)
    # bootstrap
    rng = np.random.default_rng(seed)
    n = len(jaccard)
    rhos = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        rhos[b] = spearmanr(jaccard[idx], margins[idx]).correlation
    lo, hi = np.nanpercentile(rhos, [100*alpha/2, 100*(1-alpha/2)])
    return rho_hat, (lo, hi)

def bootstrap_ci_los_q1q4(
    jaccard: np.ndarray,
    margins: np.ndarray,
    B: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
    q: float = 0.25,
):
    """Percentile CI for LOS_sensitivity (Q1 vs Q4 gap)."""
    jaccard = np.asarray(jaccard, float)
    margins = np.asarray(margins, float)
    # point estimate
    los_hat = los_q1_q4_gap(jaccard, margins, q=q)
    # bootstrap
    rng = np.random.default_rng(seed)
    n = len(jaccard)
    gaps = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        gaps[b] = los_q1_q4_gap(jaccard[idx], margins[idx], q=q)
    # drop NaNs if a resample had empty Q1/Q4 (rare with small n or many ties)
    gaps = gaps[~np.isnan(gaps)]
    lo, hi = np.percentile(gaps, [100*alpha/2, 100*(1-alpha/2)]) if gaps.size else (np.nan, np.nan)
    return los_hat, (lo, hi)

def split_by_median(scores):
    """
    Returns (threshold, labels) where labels are 0 for 'low' and 1 for 'high'.
    For even n: exactly equal splits.
    For odd n: one extra in 'high'.
    """
    n = len(scores)
    ranked = sorted(enumerate(scores), key=lambda x: x[1])  # [(idx, score), ...] ascending
    k = n // 2  # size of low split

    lows = ranked[:k]
    highs = ranked[k:]

    # labels in original order
    labels = [None] * n
    for i, _ in lows:  labels[i] = 0  # low
    for i, _ in highs: labels[i] = 1  # high

    # a midpoint threshold between the two middle values
    if n >= 2 and k > 0 and k < n:
        thr = (lows[-1][1] + highs[0][1]) / 2.0
    else:
        thr = scores[0] if n else 0.0

    return thr, labels