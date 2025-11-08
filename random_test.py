from get_token import get_token
from dataclasses import dataclass
import json
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from model_files.helper_functions import _tokens_no_entities
def get_max_overlap(
        sentence: str,
        distractor_list: List[str]
    ) :

        query_tokens = Counter(_tokens_no_entities(sentence))
        best: Optional[str] = None
        best_key = (-1, -1.0)  # (overlap_count, jaccard)
        best_scores: Dict[str, float] = {
            "overlap_count": 0,
            "jaccard": 0.0,
            "candidate_len": 0,
            "query_len": sum(query_tokens.values()),
        }

        if not query_tokens:
            return None, best_scores

        for cand in distractor_list:
            cand_tokens = Counter(_tokens_no_entities(cand))
            if not cand_tokens:
                continue

            inter = query_tokens & cand_tokens  # multiset intersection
            union = query_tokens | cand_tokens  # multiset union

            overlap_count = sum(inter.values())
            union_size = sum(union.values())
            jaccard = overlap_count / union_size if union_size else 0.0

            key = (overlap_count, jaccard)
            if key > best_key:
                best_key = key
                best = cand
                best_scores = {
                    "overlap_count": overlap_count,
                    "jaccard": jaccard,
                    "candidate_len": sum(cand_tokens.values()),
                    "query_len": sum(query_tokens.values()),
                }

        return best, best_scores
def load_model(model_name="gemma",access_token=None,device="auto"):
    # model_name = "google/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 if needed
        device_map=device,
        token=access_token,  # Use access token if required
    )
    model.eval()
    return tokenizer, model
# expects: load_model(args.model_type, access_token, device), build_batched_pt_inputs(tokenizer, texts, device)
# expects: nethook.TraceDict
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
def _mean_pool_nonpad(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # hidden: [B,T,H], attn_mask: [B,T]
    m = attn_mask.unsqueeze(-1)  # [B,T,1]
    summed = (hidden * m).sum(dim=1)
    denom = m.sum(dim=1).clamp_min(1)
    return summed / denom

def _get_cfg_heads(model) -> Tuple[int, int]:
    n_heads = int(model.config.num_attention_heads)
    head_dim = model.config.hidden_size // n_heads
    return n_heads, head_dim

def _get_o_proj_modules(model) -> List[nn.Module]:
    # HF Gemma/LLaMA-style: model.model.layers[i].self_attn.o_proj
    mods = []
    for i, layer in enumerate(model.model.layers):
        mods.append(layer.self_attn.o_proj)
    return mods

def compute_lex_scores_one_layer(
    args,
    access_token,
    anchors: List[str],
    paraphrases: List[str],
    distractors: List[str],
    layer_idx: int,
    device: str = "cuda",
):
    """
    Returns a DataFrame with Lex scores per head for a single layer.

    Lex_{ℓ,h} = s_ℓ(a,d) - s_ℓ^{d←p}(a,d),
    where we patch the distractor's pre-o_proj slice for head h at layer ℓ
    with the paraphrase's cached slice.
    """
    tokenizer, model = load_model(args.model_type, access_token, device=device)
    model.eval()
    n_heads, head_dim = _get_cfg_heads(model)
    o_projs = _get_o_proj_modules(model)
    H_total = n_heads * head_dim

    # ---------- 1) Baseline runs: cache paraphrase pre-o_proj at layer ℓ; get A/D pooled at layer ℓ ----------
    paraphrase_pre = {}  # will store the raw input to o_proj at layer ℓ: [B,T,H_total]
    distractor_pre = {}  # (optional, not required for patching but useful for checks)

    def _make_forward_pre_capture(target_store: dict):
        def hook(module, inputs):
            x = inputs[0]  # [B,T,H_total]
            target_store['val'] = x.detach().clone()
            return None
        return hook

    # paraphrase run (to capture donor slice)
    par_inputs = build_batched_pt_inputs(tokenizer, paraphrases, device=device)
    handle_par = o_projs[layer_idx].register_forward_pre_hook(_make_forward_pre_capture(paraphrase_pre))
    with nethook.TraceDict(model, [f"model.layers.{layer_idx}"]), torch.no_grad():
        _ = model(input_ids=par_inputs["input_ids"], attention_mask=par_inputs["attention_mask"], output_hidden_states=True)
    handle_par.remove()
    P_mask = par_inputs["attention_mask"]  # [B,T]
    donor_pre = paraphrase_pre['val']      # [B,T,H_total]

    # anchor & distractor baseline embeddings at layer ℓ
    anc_inputs = build_batched_pt_inputs(tokenizer, anchors, device=device)
    dis_inputs = build_batched_pt_inputs(tokenizer, distractors, device=device)

    with nethook.TraceDict(model, [f"model.layers.{layer_idx}"]) as ret, torch.no_grad():
        _ = model(input_ids=anc_inputs["input_ids"], attention_mask=anc_inputs["attention_mask"], output_hidden_states=True)
        A_hs = ret[f"model.layers.{layer_idx}"].output[0]  # [B,T,H]
        A_emb = _mean_pool_nonpad(A_hs, anc_inputs["attention_mask"])  # [B,H]

    with nethook.TraceDict(model, [f"model.layers.{layer_idx}"]) as ret, torch.no_grad():
        _ = model(input_ids=dis_inputs["input_ids"], attention_mask=dis_inputs["attention_mask"], output_hidden_states=True)
        D_hs = ret[f"model.layers.{layer_idx}"].output[0]  # [B,T,H]
        D_emb = _mean_pool_nonpad(D_hs, dis_inputs["attention_mask"])  # [B,H]

    # baseline s_ℓ(a,d) per example
    sim_AD_base = torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(A_emb, dim=-1),
        torch.nn.functional.normalize(D_emb, dim=-1),
        dim=-1
    )  # [B]

    # ---------- 2) Per-head patching: D <- P at pre-o_proj (layer ℓ) ----------
    # We'll attach a forward_pre hook that overwrites only the chosen head slice.
    def _make_head_patch_hook(head_idx: int):
        s = head_idx * head_dim
        e = (head_idx + 1) * head_dim
        donor = donor_pre  # [B,T,H_total] from paraphrase run

        def hook(module, inputs):
            x = inputs[0]  # [B,T,H_total]
            # Patch only up to min token length (usually same shapes due to batching)
            T = min(x.size(1), donor.size(1))
            x = x.clone()
            x[:, :T, s:e] = donor[:, :T, s:e].to(x.device)
            return (x,)
        return hook

    results = []
    for h in tqdm(range(n_heads), desc=f"Layer {layer_idx} heads"):
        handle = o_projs[layer_idx].register_forward_pre_hook(_make_head_patch_hook(h))
        # forward on *distractor* only; anchor stays baseline
        with nethook.TraceDict(model, [f"model.layers.{layer_idx}"]) as ret, torch.no_grad():
            _ = model(input_ids=dis_inputs["input_ids"], attention_mask=dis_inputs["attention_mask"], output_hidden_states=True)
            Dp_hs = ret[f"model.layers.{layer_idx}"].output[0]  # [B,T,H]
            Dp_emb = _mean_pool_nonpad(Dp_hs, dis_inputs["attention_mask"])  # [B,H]
        handle.remove()

        sim_AD_patched = torch.nn.functional.cosine_similarity(
            torch.nn.functional.normalize(A_emb, dim=-1),
            torch.nn.functional.normalize(Dp_emb, dim=-1),
            dim=-1
        )  # [B]

        # Lex score = mean over batch
        lex = (sim_AD_base - sim_AD_patched).mean().item()
        results.append({"layer": layer_idx, "head": h, "Lex": lex})

    df = pd.DataFrame(results)
    return df



@dataclass
class Args:
    model_type: str  # e.g., "google/gemma-3-12b" or your local HF id

args = Args(model_type="google/gemma-3-12b-pt")   # or whatever id your load_model expects
with open("/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/counterfact/Counterfact_OpenAI.jsonl", 'r') as jsonl_file:
          lines = jsonl_file.readlines()
        
dataset = [json.loads(line) for line in lines]

anchors, paraphrases, distractors = [], [], []
for record in dataset:

    distractor_list=[]
    distractor_list.extend(record["neighborhood_prompts_high_sim"])
    distractor_list.extend(record["neighborhood_prompts_low_sim"])
    distractor,_=get_max_overlap(record["edited_prompt"][0],distractor_list)
    anchors.append(record["edited_prompt"][0])
    paraphrases.append(record["edited_prompt_paraphrases_processed_testing"])
    distractors.append(distractor)

df_lex = compute_lex_scores_one_layer(
    args=args,
    access_token=get_token(),   # or None if not needed in your env
    anchors=anchors,
    paraphrases=paraphrases,
    distractors=distractors,
    layer_idx=40,
    device="cuda"
)
print(df_lex.sort_values("Lex", ascending=False).head(10))