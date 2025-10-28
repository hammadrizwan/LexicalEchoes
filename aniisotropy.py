import os,sys
sys.path.append('/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/layer_by_layer/experiments/')
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nethook
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
seed_everything(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

# MAX_TOKENS = 200_000
# PAIRS_SAMPLED = 50_000
# EXCLUDE_SPECIALS = True
EPS = 1e-12
EPS = 1e-12
PAIRS_SAMPLED = 100_000        # MC pairs per layer for baseline
MAX_TOKENS_PER_LAYER = 500_000 # reservoir cap per layer for baseline stability
EXCLUDE_SPECIALS = True
SKIP_SINGLE_TOKEN_SEQS = True  # optional: avoid seqs with <=1 valid tokens

access_token="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"

def load_model(model_name="gemma",access_token=None,device="auto"):
    model_name = "google/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 if needed
        device_map=device,
        token=access_token,  # Use access token if required
    )
    model.eval()
    return tokenizer, model
def build_batched_pt_inputs(tokenizer, texts, device="auto"):
    """
    Build a batch of prompts for Gemma-PT (base model) using plain tokenization.
    `texts` is a list[str]. If you were lowercasing upstream for IT, keep it for parity.
    Returns: dict with input_ids, attention_mask on device.
    """
    enc = tokenizer(
        [t.lower() for t in texts],   # keep/lift if you *don’t* want lowercase
        padding=True,                 # important for batching
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def is_not_wikipedia_heading(example):
        return not (example["text"].strip().startswith("=") and example["text"].strip().endswith("="))

def compute_intra_sentence_similarity_no_mask(embeddings, mask, mean_norm, eps=1e-12, average_over_batch=False):
    """
    embeddings: [L,B,T,H] already zeroed where padding (use mask_bt3 outside)
    attn_mask_bt: [B,T]
    mean_norm: [L,B,H] (L2-normalized)
    """
     # Normalize token embeddings (safe even with zeroed padding)
    emb_norm = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(eps)  # [L, B, T, H]

    # Expand mean embedding to match token dimension
    mean_norm_exp = mean_norm.unsqueeze(2)  # [L, B, 1, H]

    # Cosine similarities for each token
    cos_sim = (emb_norm * mean_norm_exp).sum(dim=-1)  # [L, B, T]

    # Average across valid tokens
    valid_counts = mask.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    # print("valid_counts",valid_counts.transpose(0, 1))
    intra_sim = cos_sim.sum(dim=2) / valid_counts.transpose(0, 1)  # [L, B]
    # print("intra_sim",intra_sim)
    # Optionally average across batch
    if average_over_batch:
        intra_sim = intra_sim.mean(dim=1)  # [L]

    return intra_sim

@torch.no_grad()
def monte_carlo_baseline_cosine(X, pairs=PAIRS_SAMPLED):
    """
    X: [N,H] L2-normalized
    """
    if X.size(0) < 2:
        return float('nan')
    N = X.size(0)
    i1 = torch.randint(0, N, (pairs,))
    i2 = torch.randint(0, N, (pairs,))
    sims = (X[i1] * X[i2]).sum(dim=-1)  # cosine because normalized
    return sims.mean().item()


def build_valid_mask(attn_mask_bt, input_ids_bt, tokenizer, exclude_special=True):
    # attn_mask_bt: [B,T] (1 valid, 0 pad)
    valid = attn_mask_bt.bool()
    if exclude_special and getattr(tokenizer, "all_special_ids", None):
        special_ids = torch.tensor(tokenizer.all_special_ids, device=input_ids_bt.device)
        not_special = ~torch.isin(input_ids_bt, special_ids)
        valid = valid & not_special
    return valid  # [B,T] bool




device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer, model = load_model(model_name="gemma-3-12b-pt",access_token=access_token,device="auto")
print("model",model)

dataset = load_dataset("wikitext", 'wikitext-103-v1')['train']
num_samples = min(10000, len(dataset))
dataset = dataset.select(range(num_samples))
dataset = dataset.filter(is_not_wikipedia_heading) # filter out headings
min_length=5
max_length=512
dataset = dataset.filter(lambda x: len(x['text']) >= 2*min_length) # filter out the frequent blank/small examples in the dataset
if max_length is not None:
    dataset = dataset.filter(lambda x: len(x['text']) <= 2*max_length)

dataloader_wikitext = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
layer_names=[]
for index in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]:
    layer_names.append(f"model.language_model.layers.{index}")
LAYER_COUNT = len(layer_names)

# running sums for dataset-mean intra-sim (per layer) — keep on CPU
intra_sum = torch.zeros(LAYER_COUNT, dtype=torch.float64)   # CPU
intra_cnt = torch.zeros(LAYER_COUNT, dtype=torch.int64)     # CPU

# token reservoirs for baseline (CPU to save VRAM)
reservoir_by_layer = [ [] for _ in range(LAYER_COUNT) ]
reservoir_sizes = [0 for _ in range(LAYER_COUNT)]


# -------------------- main pass over dataset --------------------
# model.eval()
with nethook.TraceDict(model, layer_names) as ret:
    for batch in tqdm(dataloader_wikitext, desc="Processing batches", leave=True):
        texts = batch["text"]

        inputs = build_batched_pt_inputs(
            tokenizer,
            texts,
            device=model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # forward pass (on GPU)
        _ = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )

        # retrieve outputs
        attn_mask_bt = inputs["attention_mask"]
        mask_bt3 = attn_mask_bt.unsqueeze(-1)

        # collect all traced outputs
        batch_texts_embeddings = torch.stack([ret[ln].output[0].detach().cpu() for ln in layer_names], dim=0)
        attn_mask_bt = attn_mask_bt.detach().cpu()
        mask_bt3 = mask_bt3.detach().cpu()

        # zero paddings for arithmetic
        layer_acts_masked = batch_texts_embeddings * mask_bt3.unsqueeze(0)

        valid_counts_b = attn_mask_bt.sum(dim=1)
        if SKIP_SINGLE_TOKEN_SEQS:
            keep_b = (valid_counts_b > 1)
        else:
            keep_b = torch.ones_like(valid_counts_b, dtype=torch.bool)

        if keep_b.any():
            vc = valid_counts_b[keep_b].clamp(min=1)
            mean = layer_acts_masked[:, keep_b].sum(dim=2) / vc.unsqueeze(0).unsqueeze(-1)
            mean_norm = torch.nn.functional.normalize(mean, p=2, dim=-1, eps=EPS)

            intra_sim = compute_intra_sentence_similarity_no_mask(
                embeddings=layer_acts_masked[:, keep_b],
                mask=attn_mask_bt[keep_b],
                mean_norm=mean_norm,
                eps=EPS
            )

            intra_sum += intra_sim.double().sum(dim=1).cpu()
            intra_cnt += torch.full((LAYER_COUNT,), intra_sim.size(1), dtype=torch.int64)

        # accumulate baseline tokens (already on CPU)
        valid_bt = build_valid_mask(attn_mask_bt, inputs["input_ids"].cpu(), tokenizer, exclude_special=EXCLUDE_SPECIALS)
        flat_valid = valid_bt.reshape(-1)
        H = batch_texts_embeddings.size(-1)

        for li, ln in enumerate(layer_names):
            X_bth = batch_texts_embeddings[li]
            flat = X_bth.reshape(-1, H)
            sel = flat[flat_valid]
            if sel.numel() == 0:
                continue
            sel = sel / sel.norm(dim=-1, keepdim=True).clamp_min(EPS)
            need = MAX_TOKENS_PER_LAYER - reservoir_sizes[li]
            if need <= 0:
                continue
            if sel.size(0) > need:
                idx = torch.randperm(sel.size(0))[:need]
                sel = sel[idx]
            reservoir_by_layer[li].append(sel)
            reservoir_sizes[li] += sel.size(0)

        # ---- cleanup to free VRAM ----
        del inputs, batch_texts_embeddings, layer_acts_masked, mean, mean_norm, intra_sim
        torch.cuda.empty_cache()
     

# -------------------- finalize dataset metrics --------------------
# dataset mean intra-sim per layer (CPU tensors already)
intra_mean = (intra_sum / intra_cnt.clamp(min=1)).to(torch.float32)  # [L] CPU

# compute baselines per layer from reservoirs
baselines = []
for li in range(LAYER_COUNT):
    if reservoir_by_layer[li]:
        X = torch.cat(reservoir_by_layer[li], dim=0)  # [N,H], normalized, CPU
        b = monte_carlo_baseline_cosine(X, pairs=min(PAIRS_SAMPLED, max(2, X.size(0))))
    else:
        b = float("nan")
    baselines.append(b)
baselines = torch.tensor(baselines, dtype=torch.float32)            # [L] CPU

# adjusted = raw - baseline (per layer)
adjusted = intra_mean - baselines

print("Per-layer raw intra-sim:", intra_mean.tolist())
print("Per-layer baseline:", baselines.tolist())
print("Per-layer adjusted:", adjusted.tolist())

# -------------------- plot --------------------
xs = list(range(LAYER_COUNT))
plt.figure()
plt.plot(xs, intra_mean.numpy(), label="Raw intra-sim")
plt.plot(xs, baselines.numpy(), label="Baseline (random tokens)")
plt.plot(xs, adjusted.numpy(), label="Adjusted (raw - baseline)")
plt.xlabel("Layer index (trace order)")
plt.ylabel("Cosine similarity")
plt.title("Intra-sentence similarity vs. anisotropy baseline (dataset-level)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("intra_similarity_plot.png", dpi=300)  # or .pdf/.svg if you prefer
plt.show()

#

# model.eval()
# with nethook.TraceDict(model, layer_names) as ret:
#     for batch in dataloader_wikitext:
#         batch_texts = batch['text']
#         BATCH=batch['text']
#         for text in batch_texts:
#             print("text",text+"\n")
        
#         inputs = build_batched_pt_inputs(tokenizer, batch_texts, device=device)
#         _ = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             output_hidden_states=True
#         ) 
#         batch_texts_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
#         print(batch_texts_embeddings.shape)
#         mask = inputs["attention_mask"].unsqueeze(-1)
#         print("mask",mask.shape)
        
#         layer_acts_masked = batch_texts_embeddings * mask # [L,B,T,H]
#         # lengths = mask.sum(dim=1)                                  # (B, L, 1)
#         # anchor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
#         print("layer_acts_masked",layer_acts_masked.shape)
#         # sentence mean per layer/batch (divide by valid counts)
#         valid_counts = mask.sum(dim=1).clamp(min=1)                     # [B]
#         print(valid_counts)
#         mean = layer_acts_masked.sum(dim=2) / valid_counts # [L,B,H]
#         print("mean",mean.shape)
#         # normalize mean
    
#         mean_norm = torch.nn.functional.normalize(mean, p=2, dim=-1, eps=1e-12)  # [L,B,H]

#         # IntraSim (no internal masking)
#         intra_sim = compute_intra_sentence_similarity_no_mask(
#             embeddings=layer_acts_masked,
#             mask=mask,
#             mean_norm=mean_norm,
#             eps=1e-12,
#             average_over_batch=False
#         )  # [L,B]
#         print("intra_sim",intra_sim)
        
#         # --- Baseline (random token cosine) from this layer ---
#         # (Collect over dataloader separately for better estimate; here is per-batch sketch)
#         valid_bt = build_valid_mask(inputs["attention_mask"] , inputs["input_ids"], tokenizer, exclude_special=EXCLUDE_SPECIALS)
#         flat = batch_texts_embeddings[0].reshape(-1, batch_texts_embeddings.size(-1))    # [B*T, H] (use non-masked to avoid bias)
#         sel  = flat[valid_bt.reshape(-1)]
#         sel  = sel / sel.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         if sel.size(0) >= 2:
#             baseline = monte_carlo_baseline_cosine(sel.cpu(), pairs=min(PAIRS_SAMPLED, sel.size(0)))
#         else:
#             baseline = float('nan')

#         # adjusted intra-sim (example): subtract baseline per layer
#         intra_sim_adjusted = intra_sim - baseline

#         print("IntraSim raw:", intra_sim.mean().item(), "Baseline:", baseline, "Adjusted:", intra_sim_adjusted.mean().item())   
#         break
