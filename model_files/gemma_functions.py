
import torch, json, linecache, os
dir_path = os.path.dirname(os.path.abspath(__file__))
import nethook
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from visualization_quora_paws import analyze_and_save_distances
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from helper_functions import apply_whitening_batch,bootstrap_ci_spearman,bootstrap_ci_los_q1q4,_jaccard_overlap_pct,_counts
from scipy.stats import spearmanr
_spearman = lambda x, y: spearmanr(x, y, nan_policy="omit")
from collections import defaultdict
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
PROMPT_AVG="""Act as a sentence embedding model for STS.
Distribute the global meaning across content tokens so mean pooling over user tokens captures entities, numbers, dates/times, locations, modality, negation, conditionals, and event order. Ignore headers and formatting."""
# 
# """
# You are a sentence embedding model for Semantic Textual Similarity (STS).
# Your job is to produce token-level hidden states whose pooled vector encodes the full meaning of the input text so that semantically equivalent texts are close and non-equivalent texts are far in embedding space.

# Objectives
# - Optimize representations for **semantic equivalence**, not lexical overlap or style.
# - Similarity should reflect **truth-conditional meaning**: entities, quantities, dates/times, locations, modality, negation, conditionals, and event order.
# - Be invariant to superficial wording: tense changes, punctuation, synonyms, paraphrase, formatting, casing, and stopwords.

# Sensitivity (what must change the embedding)
# - Changes to entities, numbers, dates/times, locations.
# - Polarity (negation), modality (may/must), hedges, and conditionals.
# - Causal/event order or scope changes; added/removed facts.

# Pooling-aware guidance
# - The embedding is read via **mean pooling over tokens**, distribute the global meaning across **content tokens**, and keep format/role/special tokens minimally informative.

# Robustness
# - Avoid encoding prompt boilerplate or role/formatting tokens.
# - Downweight spurious lexical cues and repeated surface forms.
# - Keep representations stable across paraphrases and minor rephrasings.

# Output behavior
# - Do not emit explanations or extra text; focus on producing informative hidden states suitable for pooling into a single semantic vector."""

PROMPT_LASTTOKEN="""
You are a sentence embedding model for Semantic Textual Similarity (STS).
Your job is to produce token-level hidden states whose pooled vector encodes the full meaning of the input text so that semantically equivalent texts are close and non-equivalent texts are far in embedding space.

Objectives
- Optimize representations for **semantic equivalence**, not lexical overlap or style.
- Similarity should reflect **truth-conditional meaning**: entities, quantities, dates/times, locations, modality, negation, conditionals, and event order.
- Be invariant to superficial wording: tense changes, punctuation, synonyms, paraphrase, formatting, casing, and stopwords.

Sensitivity (what must change the embedding)
- Changes to entities, numbers, dates/times, locations.
- Polarity (negation), modality (may/must), hedges, and conditionals.
- Causal/event order or scope changes; added/removed facts.

Pooling-aware guidance
- The embedding is read from the **last token**, place a compact **global summary** of the sentence meaning at the final position; avoid over-weighting recent/local tokens.

Robustness
- Avoid encoding prompt boilerplate or role/formatting tokens.
- Downweight spurious lexical cues and repeated surface forms.
- Keep representations stable across paraphrases and minor rephrasings.

Output behavior
- Do not emit explanations or extra text; focus on producing informative hidden states suitable for pooling into a single semantic vector."""
#----------------------------------------------------------------------------
# Section: Gemma PT Counterfact
#----------------------------------------------------------------------------
#region Gemma PT Counterfact

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

def compute_intra_sentence_similarity(embeddings, mask, mean_norm, eps=1e-12,average_over_batch=False):
    """
    Compute IntraSimᶩ(s) = (1/n) Σ_i cos(~sᶩ, fᶩ(s,i))
    for each layer and batch.

    Args:
        embeddings: Tensor [L, B, T, H]
            Layerwise token embeddings.
        mask: Tensor [B, T, 1]
            1 for valid tokens, 0 for padding.
        mean_norm: Tensor [L, B, H]
            Normalized mean embedding per layer and batch.
        eps: float
            Small epsilon for numerical stability.

    Returns:
        intra_sim: Tensor [L, B]
            Intra-sentence similarity per layer and batch.
    """
    L, B, T, H = embeddings.shape

    # Normalize token embeddings
    emb_norm = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Expand normalized mean to match token dimension
    mean_norm_exp = mean_norm.unsqueeze(2)  # [L, B, 1, H]

    # Cosine similarities for each token
    cos_sim = (emb_norm * mean_norm_exp).sum(dim=-1)  # [L, B, T]

    # Mask out padding
    mask_bt = mask.squeeze(-1).unsqueeze(0)           # [1, B, T]
    cos_sim = cos_sim * mask_bt

    # Average across valid tokens
    valid_counts = mask_bt.sum(dim=2).clamp(min=1)    # [1, B]
    intra_sim = cos_sim.sum(dim=2) / valid_counts     # [L, B]
    # Optionally average across batch
    if average_over_batch:
        intra_sim = intra_sim.mean(dim=1)             # [L]
    return intra_sim

def gemma_pt_counterfact_scpp(data_loader,args,access_token,layers,device="auto"):
    tokenizer,model = load_model(args.model_type,access_token,device=device)
    model.eval()
    # Example: access specific fields
    
    all_fail_flags_by_layer = defaultdict(list)
    all_viols_by_layer = defaultdict(list)
    LO_anchor_paraphrase_list_by_layer = defaultdict(list)
    LO_anchor_distractor_list_by_layer = defaultdict(list)
    LO_paraphrase_distractor_list_by_layer = defaultdict(list)
    LO_negmax_list_by_layer = defaultdict(list)
    LOc_list_by_layer = defaultdict(list)
    DISTc_list_by_layer = defaultdict(list)

    average_margin_lor_low_by_layer = defaultdict(float)
    average_margin_violation_lor_low_by_layer = defaultdict(float)
    failure_rate_lor_low_by_layer = defaultdict(int)

    average_margin_lor_high_by_layer = defaultdict(float)
    average_margin_violation_lor_high_by_layer = defaultdict(float)
    failure_rate_lor_high_by_layer = defaultdict(int)

    incorrect_pairs_by_layer = defaultdict(list)
    correct_pairs_by_layer = defaultdict(list)
    neg_pairs_by_layer = defaultdict(list)
    pos_pairs_by_layer = defaultdict(list)

    jaccard_scores_list_by_layer = defaultdict(list)
    signed_margins_by_layer = defaultdict(list)

    intrasims_dict=defaultdict(list)
    l = layers
    
    with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
   
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
              
                inputs = build_batched_pt_inputs(tokenizer, anchors, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                ) 
                anchor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
                mask = inputs["attention_mask"].unsqueeze(-1)
                if(args.mode):
                    padded_zero_embeddings=anchor_sentence_embeddings * mask
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = mask.sum(dim=1)                                  # (B, L, 1)
                    anchor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    mean_norm = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=2, eps=1e-12)
                    intrasims= compute_intra_sentence_similarity(padded_zero_embeddings, mask, mean_norm, eps=1e-12)
                    # print("intrasims",intrasims.shape)
                else:
                    L, B, T, E = anchor_sentence_embeddings.shape
                    base_mask = mask.squeeze(-1).to(anchor_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=anchor_sentence_embeddings.device, dtype=anchor_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    anchor_sentence_embeddings = anchor_sentence_embeddings.gather(2, idx).squeeze(2)
                #_________________________________________________________________________________________
                inputs = build_batched_pt_inputs(tokenizer, paraphrases, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                ) 
                paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
                mask = inputs["attention_mask"].unsqueeze(-1)
                if(args.mode):
                    padded_zero_embeddings=paraphrase_sentence_embeddings * mask
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = mask.sum(dim=1)                                     # (B, ,L, 1)
                    paraphrase_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    
                else:
                    L, B, T, E = paraphrase_sentence_embeddings.shape
                    base_mask = mask.squeeze(-1).to(paraphrase_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=paraphrase_sentence_embeddings.device, dtype=paraphrase_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    paraphrase_sentence_embeddings = paraphrase_sentence_embeddings.gather(2, idx).squeeze(2)
                #_________________________________________________________________________________________
                inputs = build_batched_pt_inputs(tokenizer, distractors, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                ) 
                distractor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
                mask = inputs["attention_mask"].unsqueeze(-1)
                if(args.mode):
                    padded_zero_embeddings=distractor_sentence_embeddings * mask
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = mask.sum(dim=1)                                     # (B, ,L, 1)
                    distractor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)

                else:
                    L, B, T, E = distractor_sentence_embeddings.shape
                    base_mask = mask.squeeze(-1).to(distractor_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=distractor_sentence_embeddings.device, dtype=distractor_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    distractor_sentence_embeddings = distractor_sentence_embeddings.gather(2, idx).squeeze(2)
                #_________________________________________________________________________________________

                anchor_sentence_embeddings = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=2, eps=1e-12)
                paraphrase_sentence_embeddings = torch.nn.functional.normalize(paraphrase_sentence_embeddings, p=2, dim=2, eps=1e-12)
                distractor_sentence_embeddings = torch.nn.functional.normalize(distractor_sentence_embeddings, p=2, dim=2, eps=1e-12)
                # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                # --- Cosine similctor   arities ---
                cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embeddings, paraphrase_sentence_embeddings, dim=2)
                cosine_anchor_distractor = F.cosine_similarity(anchor_sentence_embeddings, distractor_sentence_embeddings, dim=2)
                cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_sentence_embeddings, distractor_sentence_embeddings, dim=2)


                distances_anchor_paraphrase = torch.norm(anchor_sentence_embeddings - paraphrase_sentence_embeddings, p=2,dim=2)
            
                distances_anchor_distractor   = torch.norm(anchor_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=2)
                distances_paraphrase_distractor = torch.norm(paraphrase_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=2)

                for i in range(distances_anchor_paraphrase.size(1)):
                    anchor = anchors[i]
                    paraphrase = paraphrases[i]
                    distractor = distractors[i]
                    LOS_flag = LOS_flags[i]
                    
                    # --- text-only overlaps computed ONCE per sample ---
                    LO_ap = _jaccard_overlap_pct(_counts(anchor, "entities"), _counts(paraphrase, "entities"))
                    LO_ad = _jaccard_overlap_pct(_counts(anchor, "entities"), _counts(distractor, "entities"))
                    LO_pd = _jaccard_overlap_pct(_counts(paraphrase, "entities"), _counts(distractor, "entities"))
                    LO_negmax = max(LO_ad, LO_pd)
                    LOc = LO_negmax - LO_ap

                    for layer_idx, layer in enumerate(layers):
                        layer_key = str(layer)
                        layer_dir = os.path.join(args.save_path, layer_key)
                        os.makedirs(layer_dir, exist_ok=True)
                        file_save_path = os.path.join(layer_dir, "counterfact_results.jsonl")

                        dap = distances_anchor_paraphrase[layer_idx, i].item()
                        dpd = distances_paraphrase_distractor[layer_idx, i].item()
                        dad = distances_anchor_distractor[layer_idx, i].item()

                        cos_ap = cosine_anchor_paraphrase[layer_idx, i].item()
                        cos_ad = cosine_anchor_distractor[layer_idx, i].item()
                        cos_pd = cosine_paraphrase_distractor[layer_idx, i].item()
                        intrasim=intrasims[layer_idx,i].item()

                        condition_anchor = dad < dap
                        condition_paraphrase = dpd < dap
                        failure = condition_anchor or condition_paraphrase
                        possible_margin_violation = dap - min(dpd, dad)

                        with open(file_save_path, 'a') as jsonl_file_writer:
                            json.dump({
                                "layer": layer_key,
                                "distance_failure": failure,
                                "lexical_overlap_flag": LOS_flag,
                                "similarity_failure": ((cos_ad > cos_ap) or (cos_pd > cos_ap)),
                                "distances": {"dist_cap1_cap2": dap, "dist_cap1_neg": dad, "dist_cap2_neg": dpd},
                                "similarities": {"cos_cap1_cap2": cos_ap, "cos_cap1_neg": cos_ad, "cos_cap2_neg": cos_pd},
                                "anchor": anchor, "paraphrase": paraphrase, "distractor": distractor,
                                "score_jaccard": scores_jaccard[i],
                                "score_overlap": scores_overlap[i],
                                "score_containment": scores_containment[i],
                                "LO_ap": LO_ap, "LO_ad": LO_ad, "LO_pd": LO_pd,
                                "LO_negmax": LO_negmax, "LOc": LOc
                            }, jsonl_file_writer)
                            jsonl_file_writer.write("\n")

                        # --- per-layer metric collectors ---
                        intrasims_dict[layer_key].append(intrasim)
                        all_fail_flags_by_layer[layer_key].append(1 if failure else 0)
                        all_viols_by_layer[layer_key].append(max(0.0, -possible_margin_violation))
                        LO_anchor_paraphrase_list_by_layer[layer_key].append(LO_ap)
                        LO_anchor_distractor_list_by_layer[layer_key].append(LO_ad)
                        LO_paraphrase_distractor_list_by_layer[layer_key].append(LO_pd)
                        LO_negmax_list_by_layer[layer_key].append(LO_negmax)
                        LOc_list_by_layer[layer_key].append(LOc)
                        DISTc_list_by_layer[layer_key].append(possible_margin_violation)

                        if (LOS_flag == "low"):
                            if failure:
                                jaccard_scores_list_by_layer[layer_key].append(scores_jaccard[i])
                                signed_margins_by_layer[layer_key].append(possible_margin_violation)
                                average_margin_violation_lor_low_by_layer[layer_key] += possible_margin_violation
                                failure_rate_lor_low_by_layer[layer_key] += 1
                            average_margin_lor_low_by_layer[layer_key] += possible_margin_violation
                        elif (LOS_flag == "high"):
                            if failure:
                                jaccard_scores_list_by_layer[layer_key].append(scores_jaccard[i])
                                signed_margins_by_layer[layer_key].append(possible_margin_violation)
                                average_margin_violation_lor_high_by_layer[layer_key] += possible_margin_violation
                                failure_rate_lor_high_by_layer[layer_key] += 1
                            average_margin_lor_high_by_layer[layer_key] += possible_margin_violation

                        if (dad <= dpd):
                            incorrect_pairs_by_layer[layer_key].append(dad)
                            neg_pairs_by_layer[layer_key].append((anchor, distractor))
                        else:
                            incorrect_pairs_by_layer[layer_key].append(dpd)
                            neg_pairs_by_layer[layer_key].append((paraphrase, distractor))

                        correct_pairs_by_layer[layer_key].append(dap)
                        pos_pairs_by_layer[layer_key].append((anchor, paraphrase))


            for layer in tqdm(layers, desc="Processing Layers Final"):
                layer_key = str(layer)
                layer_dir = os.path.join(args.save_path, layer_key)
                os.makedirs(layer_dir, exist_ok=True)
                summary_path = os.path.join(layer_dir, "counterfact_summary.jsonl")

                # numpy conversions
                LO_negmax_arr = np.asarray(LO_negmax_list_by_layer[layer_key], dtype=float)
                all_fail_arr = np.asarray(all_fail_flags_by_layer[layer_key], dtype=int)
                LOc_arr = np.asarray(LOc_list_by_layer[layer_key], dtype=float)
                DISTc_arr = np.asarray(DISTc_list_by_layer[layer_key], dtype=float)
                all_viols_arr = np.asarray(all_viols_by_layer[layer_key], dtype=float)
                intrasims_dict_arr = np.asarray(intrasims_dict[layer_key], dtype=float).mean()
                # print("intrasims_dict_arr",intrasims_dict_arr.shape)
                # failure stats
                total_samples_layer = len(all_fail_arr)
                total_failures_layer = int(all_fail_arr.sum()) if total_samples_layer > 0 else 0
                failure_rate_layer = (total_failures_layer / total_samples_layer) if total_samples_layer > 0 else 0.0

                margin_violation_layer = float(np.nansum(all_viols_arr)) if all_viols_arr.size > 0 else 0.0
                avg_margin_violation_layer = (margin_violation_layer / total_failures_layer) if total_failures_layer > 0 else 0.0

                # stratified fail rates by overlap (Q1 vs Q4)
                fail_rate_high = np.nan
                fail_rate_low = np.nan
                fail_rate_gap = np.nan
                rel_risk = np.nan
                if LO_negmax_arr.size > 0 and all_fail_arr.size > 0:
                    q1, q4 = np.quantile(LO_negmax_arr, [0.25, 0.75])
                    low_bin = (LO_negmax_arr <= q1)
                    high_bin = (LO_negmax_arr >= q4)
                    if low_bin.any():
                        fail_rate_low = float(all_fail_arr[low_bin].mean())
                    if high_bin.any():
                        fail_rate_high = float(all_fail_arr[high_bin].mean())
                    if low_bin.any() and high_bin.any():
                        fail_rate_gap = float(fail_rate_high - fail_rate_low)
                        rel_risk = float(fail_rate_high / max(fail_rate_low, 1e-12))

                # predictive power (AUC)
                auc_overlap = np.nan
                if all_fail_arr.size > 0 and (all_fail_arr.min() != all_fail_arr.max()):
                    try:
                        auc_overlap = float(roc_auc_score(all_fail_arr, LOc_arr))
                    except Exception:
                        auc_overlap = np.nan

                # failure severity gap (within failures)
                sev_gap = np.nan
                mask_fail = (all_fail_arr == 1)
                if mask_fail.any():
                    Jf = LO_negmax_arr[mask_fail]
                    Vf = all_viols_arr[mask_fail]
                    if Jf.size > 0:
                        jf_q1, jf_q4 = np.quantile(Jf, [0.25, 0.75])
                        lo_f = Vf[Jf <= jf_q1]
                        hi_f = Vf[Jf >= jf_q4]
                        if lo_f.size > 0 and hi_f.size > 0:
                            sev_gap = float(np.median(hi_f) - np.median(lo_f))

                with open(summary_path, 'a') as jsonl_file_writer:
                    json.dump({
                        "layer": layer_key,
                        "intrasims_dict": float(intrasims_dict_arr),
                        "Failure Rate": failure_rate_layer,
                        "Average Margin Violation": avg_margin_violation_layer,
                        "FailRate_high_Q4": None if np.isnan(fail_rate_high) else fail_rate_high,
                        "FailRate_low_Q1":  None if np.isnan(fail_rate_low)  else fail_rate_low,
                        "FailRate_gap_Q4_minus_Q1": None if np.isnan(fail_rate_gap) else fail_rate_gap,
                        "RelativeRisk_high_over_low": None if np.isnan(rel_risk) else rel_risk,
                        "AUC_LOcontrast_to_failure": None if np.isnan(auc_overlap) else auc_overlap,
                        "Failure_severity_median_gap_Q4_minus_Q1": None if np.isnan(sev_gap) else sev_gap
                    }, jsonl_file_writer)
                    jsonl_file_writer.write("\n")

                # optional: per-layer plots
                _ = analyze_and_save_distances(
                    incorrect_pairs_by_layer[layer_key],
                    correct_pairs_by_layer[layer_key],
                    title_prefix=f"Counterfact_Layer_{layer_key}",
                    out_dir=layer_dir,
                    group_neg_name="Non-Paraphrase",
                    group_pos_name="Paraphrase",
                    neg_text_pairs=neg_pairs_by_layer[layer_key],
                    pos_text_pairs=pos_pairs_by_layer[layer_key],
                    neg_color="#2EA884",
                    pos_color="#D7732D",
                    hist_alpha=0.35,
                    kde_linewidth=2.0,
                    ecdf_linewidth=2.0,
                    tau_color="#000000",
                    tau_linestyle="--",
                    tau_linewidth=1.5,
                    violin_facealpha=0.35,
                    box_facealpha=0.35,
                )
#endregion         


#----------------------------------------------------------------------------
# Section: Gemma IT Counterfact
#----------------------------------------------------------------------------
#region Gemma IT Counterfact

def get_mask_text(inputs, tokenizer, texts, embeddings, device="auto"):
    """
    If embeddings.ndim == 3: (B, L, H)  -> returns (B, L, 1)
    If embeddings.ndim == 4: (T, B, L, H) -> returns (T, B, L, 1)
    """
    if device == "auto":
        device = embeddings.device

    # Tokenize raw texts WITHOUT specials to get the content pattern
    _tok = tokenizer(list(texts), add_special_tokens=False)

    if embeddings.ndim == 3:
        B, L, _H = embeddings.shape
        T = None
    elif embeddings.ndim == 4:
        T, B, L, _H = embeddings.shape
    else:
        raise ValueError(f"embeddings must be 3D or 4D, got shape {embeddings.shape}")

    # Build a base (B, L, 1) mask once
    base_mask = torch.zeros((B, L, 1), dtype=embeddings.dtype, device=device)

    input_ids_batch = inputs["input_ids"]      # (B, L)
    attention_mask  = inputs["attention_mask"] # (B, L)

    for i, pat in enumerate(_tok["input_ids"]):
        seq = input_ids_batch[i].tolist()
        m = len(pat)
        best_j = -1
        if m > 0 and m <= len(seq):
            # reverse search: stop at the first match from the end
            for j in range(len(seq) - m, -1, -1):
                if seq[j:j+m] == pat:
                    best_j = j
                    break

        if best_j >= 0:
            base_mask[i, best_j:best_j+m, 0] = 1.0
        else:
            # Fallback: non-empty mask using attention
            base_mask[i] = attention_mask[i].unsqueeze(-1).type_as(base_mask)

    # Safety: guarantee at least 1 token selected per row
    empty = (base_mask.sum(dim=1) == 0).squeeze(-1)
    if empty.any():
        base_mask[empty] = attention_mask[empty].unsqueeze(-1).type_as(base_mask)

    # If 4D embeddings: replicate across layers to (T, B, L, 1)
    if T is not None:
        # Use expand, not repeat, to avoid extra memory unless you need a writeable copy
        mask = base_mask.unsqueeze(0).expand(T, B, L, 1)
    else:
        mask = base_mask

    return mask

def build_batched_chat_inputs(tokenizer, texts, add_generation_prompt=True,PROMPT="", device="auto"):
    """
    Build a batch of chat-formatted prompts for Gemma-IT using tokenizer.apply_chat_template.
    `texts` is a list[str] (already lowercased / formatted upstream if you want).
    Returns: dict with input_ids, attention_mask on device.
    """
    # Each item in the batch is a full conversation (list of messages)
    conversations = [
        [
            
            {"role": "user",
             "content": [{"type": "text", "text": "{}".format(t.lower())}]}
        ]
        for t in texts
    ]
    # conversations = [
    # [
    #     {"role": "system",
    #      "content": [{"type": "text", "text": PROMPT.strip()}]},
    #     {"role": "user",
    #      "content": [{"type": "text", "text": "{}".format(t.lower())}]}
    # ]
#     for t in texts
# ]

    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        padding=True,                # important for batching
        return_dict=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in inputs.items()}

def compute_intrasim_same_mask_across_layers(
    embeddings: torch.Tensor,           # [L, B, T, H]  unmasked token embeddings
    mask: torch.Tensor,                 # [B, T, 1] or [L, B, T, 1], same across layers
    average_over_batch: bool = False,   # True -> returns [L]; False -> returns [L, B]
    eps: float = 1e-12,
):
    """
    Computes IntraSimᶩ(s) = (1/n) Σ_i cos(~sᶩ, fᶩ(s,i)) per layer, with masking.
    The mask is assumed to be identical across layers (one mask per sample).

    Args:
        embeddings: [L, B, T, H] layerwise token embeddings (unmasked).
        mask: [B, T, 1] or [L, B, T, 1]. If 4D, it's collapsed across L.
        average_over_batch: if True, returns [L] averaged across batch; else [L, B].
        eps: small constant for numerical stability.

    Returns:
        intra_sim: [L, B] if average_over_batch=False, else [L]
        s_mean:    [L, B, H] masked mean embedding per layer & sample (not normalized)
    """
    assert embeddings.ndim == 4, "embeddings must be [L, B, T, H]"
    L, B, T, H = embeddings.shape
    device = embeddings.device
    dtype  = embeddings.dtype

    # --- Normalize / standardize mask to [B, T, 1] ---
    if mask.ndim == 4:       # [L, B, T, 1] -> collapse layer
        # mask is identical across layers; collapse via any() or take mask[0]
        mask_bt1 = mask.any(dim=0).to(dtype=dtype)
    elif mask.ndim == 3:     # [B, T, 1]
        mask_bt1 = mask.to(dtype=dtype)
    else:
        raise ValueError("mask must be [B, T, 1] or [L, B, T, 1]")

    # Broadcast mask to embeddings for summations
    mask_lbt1 = mask_bt1.unsqueeze(0)                 # [1, B, T, 1]
    mask_lbt  = mask_bt1.squeeze(-1).unsqueeze(0)     # [1, B, T]

    # --- Masked mean per layer & sample: s_meanᶩ = (Σ_i fᶩ(s,i)) / n ---
    summed = (embeddings * mask_lbt1).sum(dim=2)      # [L, B, H]
    counts = mask_bt1.sum(dim=1).clamp(min=1)         # [B, 1]
    s_mean = summed / counts.unsqueeze(0)             # [L, B, H]

    # --- Normalize for cosine ---
    s_mean_norm = torch.nn.functional.normalize(s_mean, p=2, dim=-1, eps=eps)  # [L, B, H]
    emb_norm    = torch.nn.functional.normalize(embeddings, p=2, dim=-1, eps=eps)  # [L, B, T, H]

    # --- Cosine with each token, mask pads, average over valid tokens ---
    cos_lbt = (emb_norm * s_mean_norm.unsqueeze(2)).sum(dim=-1)   # [L, B, T]
    cos_lbt = cos_lbt * mask_lbt                                   # [L, B, T]
    valid = mask_lbt.sum(dim=2).clamp(min=1)                       # [1, B]
    intra_sim = cos_lbt.sum(dim=2) / valid                         # [L, B]

    if average_over_batch:
        intra_sim = intra_sim.mean(dim=1)                          # [L]

    return intra_sim

def gemma_it_counterfact_scpp(data_loader,args,access_token,layers,device="auto"):
    tokenizer,model = load_model(args.model_type,access_token,device=device)
    model.eval()
    # Example: access specific fields
    
    all_fail_flags_by_layer = defaultdict(list)
    all_viols_by_layer = defaultdict(list)
    LO_anchor_paraphrase_list_by_layer = defaultdict(list)
    LO_anchor_distractor_list_by_layer = defaultdict(list)
    LO_paraphrase_distractor_list_by_layer = defaultdict(list)
    LO_negmax_list_by_layer = defaultdict(list)
    LOc_list_by_layer = defaultdict(list)
    DISTc_list_by_layer = defaultdict(list)

    average_margin_lor_low_by_layer = defaultdict(float)
    average_margin_violation_lor_low_by_layer = defaultdict(float)
    failure_rate_lor_low_by_layer = defaultdict(int)

    average_margin_lor_high_by_layer = defaultdict(float)
    average_margin_violation_lor_high_by_layer = defaultdict(float)
    failure_rate_lor_high_by_layer = defaultdict(int)

    incorrect_pairs_by_layer = defaultdict(list)
    correct_pairs_by_layer = defaultdict(list)
    neg_pairs_by_layer = defaultdict(list)
    pos_pairs_by_layer = defaultdict(list)

    jaccard_scores_list_by_layer = defaultdict(list)
    signed_margins_by_layer = defaultdict(list)
    intrasims_dict=defaultdict(list)

    l = layers
    
    with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
    
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
              
                inputs = build_batched_chat_inputs(tokenizer, anchors, PROMPT_AVG, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                ) 
                anchor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
                # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                content_mask = get_mask_text(inputs, tokenizer, anchors, anchor_sentence_embeddings, device=device) 
                # print("content_mask",content_mask.shape)
                if(args.mode):
                    intrasims= compute_intrasim_same_mask_across_layers(anchor_sentence_embeddings, content_mask, eps=1e-12)
                    # print("intrasims",intrasims.shape)
                    padded_zero_embeddings=anchor_sentence_embeddings * content_mask
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = content_mask.sum(dim=2)                                     # (B, ,L, 1)
                    anchor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    mean_norm = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=2, eps=1e-12)
                    # print("mean_norm",mean_norm.shape)
                    
                    

                else:
                    L, B, T, E = anchor_sentence_embeddings.shape
                    base_mask = content_mask[0].squeeze(-1).to(anchor_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=anchor_sentence_embeddings.device, dtype=anchor_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    anchor_sentence_embeddings = anchor_sentence_embeddings.gather(2, idx).squeeze(2)
                #_________________________________________________________________________________________
                inputs = build_batched_chat_inputs(tokenizer, paraphrases, PROMPT_AVG, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                ) 
                paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
                content_mask = get_mask_text(inputs, tokenizer, paraphrases, paraphrase_sentence_embeddings, device=device) 
                if(args.mode):
                    padded_zero_embeddings=paraphrase_sentence_embeddings * content_mask
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = content_mask.sum(dim=2)                                     # (B, ,L, 1)
                    paraphrase_sentence_embeddings = sum_hidden / lengths.clamp(min=1)

                else:
                    L, B, T, E = paraphrase_sentence_embeddings.shape
                    base_mask = content_mask[0].squeeze(-1).to(paraphrase_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=paraphrase_sentence_embeddings.device, dtype=paraphrase_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    paraphrase_sentence_embeddings = paraphrase_sentence_embeddings.gather(2, idx).squeeze(2)

                    
                #_________________________________________________________________________________________
                inputs = build_batched_chat_inputs(tokenizer, distractors, PROMPT_AVG, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                ) 
                distractor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
                content_mask = get_mask_text(inputs, tokenizer, distractors, distractor_sentence_embeddings, device=device) 
                if(args.mode):
                    padded_zero_embeddings=distractor_sentence_embeddings * content_mask
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = content_mask.sum(dim=2)                                     # (B, ,L, 1)
                    distractor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)

                else:
                    L, B, T, E = distractor_sentence_embeddings.shape
                    base_mask = content_mask[0].squeeze(-1).to(distractor_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=distractor_sentence_embeddings.device, dtype=distractor_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    distractor_sentence_embeddings = distractor_sentence_embeddings.gather(2, idx).squeeze(2)
                #_________________________________________________________________________________________

                anchor_sentence_embeddings = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=2, eps=1e-12)
                paraphrase_sentence_embeddings = torch.nn.functional.normalize(paraphrase_sentence_embeddings, p=2, dim=2, eps=1e-12)
                distractor_sentence_embeddings = torch.nn.functional.normalize(distractor_sentence_embeddings, p=2, dim=2, eps=1e-12)
                # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                # --- Cosine similctor   arities ---
                cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embeddings, paraphrase_sentence_embeddings, dim=2)
                cosine_anchor_distractor = F.cosine_similarity(anchor_sentence_embeddings, distractor_sentence_embeddings, dim=2)
                cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_sentence_embeddings, distractor_sentence_embeddings, dim=2)


                distances_anchor_paraphrase = torch.norm(anchor_sentence_embeddings - paraphrase_sentence_embeddings, p=2,dim=2)
            
                distances_anchor_distractor   = torch.norm(anchor_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=2)
                distances_paraphrase_distractor = torch.norm(paraphrase_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=2)

                for i in range(distances_anchor_paraphrase.size(1)):
                    anchor = anchors[i]
                    paraphrase = paraphrases[i]
                    distractor = distractors[i]
                    LOS_flag = LOS_flags[i]

                    # --- text-only overlaps computed ONCE per sample ---
                    LO_ap = _jaccard_overlap_pct(_counts(anchor, "entities"), _counts(paraphrase, "entities"))
                    LO_ad = _jaccard_overlap_pct(_counts(anchor, "entities"), _counts(distractor, "entities"))
                    LO_pd = _jaccard_overlap_pct(_counts(paraphrase, "entities"), _counts(distractor, "entities"))
                    LO_negmax = max(LO_ad, LO_pd)
                    LOc = LO_negmax - LO_ap

                    for layer_idx, layer in enumerate(layers):
                        layer_key = str(layer)
                        layer_dir = os.path.join(args.save_path, layer_key)
                        os.makedirs(layer_dir, exist_ok=True)
                        file_save_path = os.path.join(layer_dir, "counterfact_results.jsonl")

                        dap = distances_anchor_paraphrase[layer_idx, i].item()
                        dpd = distances_paraphrase_distractor[layer_idx, i].item()
                        dad = distances_anchor_distractor[layer_idx, i].item()

                        cos_ap = cosine_anchor_paraphrase[layer_idx, i].item()
                        cos_ad = cosine_anchor_distractor[layer_idx, i].item()
                        cos_pd = cosine_paraphrase_distractor[layer_idx, i].item()
                        intrasim=intrasims[layer_idx,i].item()
                        
                        condition_anchor = dad < dap
                        condition_paraphrase = dpd < dap
                        failure = condition_anchor or condition_paraphrase
                        possible_margin_violation = dap - min(dpd, dad)

                        with open(file_save_path, 'a') as jsonl_file_writer:
                            json.dump({
                                "layer": layer_key,
                                "distance_failure": failure,
                                "lexical_overlap_flag": LOS_flag,
                                "similarity_failure": ((cos_ad > cos_ap) or (cos_pd > cos_ap)),
                                "distances": {"dist_cap1_cap2": dap, "dist_cap1_neg": dad, "dist_cap2_neg": dpd},
                                "similarities": {"cos_cap1_cap2": cos_ap, "cos_cap1_neg": cos_ad, "cos_cap2_neg": cos_pd},
                                "anchor": anchor, "paraphrase": paraphrase, "distractor": distractor,
                                "score_jaccard": scores_jaccard[i],
                                "score_overlap": scores_overlap[i],
                                "score_containment": scores_containment[i],
                                "LO_ap": LO_ap, "LO_ad": LO_ad, "LO_pd": LO_pd,
                                "LO_negmax": LO_negmax, "LOc": LOc
                            }, jsonl_file_writer)
                            jsonl_file_writer.write("\n")

                        # --- per-layer metric collectors ---
                        intrasims_dict[layer_key].append(intrasim)
                        all_fail_flags_by_layer[layer_key].append(1 if failure else 0)
                        all_viols_by_layer[layer_key].append(max(0.0, -possible_margin_violation))
                        LO_anchor_paraphrase_list_by_layer[layer_key].append(LO_ap)
                        LO_anchor_distractor_list_by_layer[layer_key].append(LO_ad)
                        LO_paraphrase_distractor_list_by_layer[layer_key].append(LO_pd)
                        LO_negmax_list_by_layer[layer_key].append(LO_negmax)
                        LOc_list_by_layer[layer_key].append(LOc)
                        DISTc_list_by_layer[layer_key].append(possible_margin_violation)

                        if (LOS_flag == "low"):
                            if failure:
                                jaccard_scores_list_by_layer[layer_key].append(scores_jaccard[i])
                                signed_margins_by_layer[layer_key].append(possible_margin_violation)
                                average_margin_violation_lor_low_by_layer[layer_key] += possible_margin_violation
                                failure_rate_lor_low_by_layer[layer_key] += 1
                            average_margin_lor_low_by_layer[layer_key] += possible_margin_violation
                        elif (LOS_flag == "high"):
                            if failure:
                                jaccard_scores_list_by_layer[layer_key].append(scores_jaccard[i])
                                signed_margins_by_layer[layer_key].append(possible_margin_violation)
                                average_margin_violation_lor_high_by_layer[layer_key] += possible_margin_violation
                                failure_rate_lor_high_by_layer[layer_key] += 1
                            average_margin_lor_high_by_layer[layer_key] += possible_margin_violation

                        if (dad <= dpd):
                            incorrect_pairs_by_layer[layer_key].append(dad)
                            neg_pairs_by_layer[layer_key].append((anchor, distractor))
                        else:
                            incorrect_pairs_by_layer[layer_key].append(dpd)
                            neg_pairs_by_layer[layer_key].append((paraphrase, distractor))

                        correct_pairs_by_layer[layer_key].append(dap)
                        pos_pairs_by_layer[layer_key].append((anchor, paraphrase))


            for layer in tqdm(layers, desc="Processing Layers Final"):
                layer_key = str(layer)
                layer_dir = os.path.join(args.save_path, layer_key)
                os.makedirs(layer_dir, exist_ok=True)
                summary_path = os.path.join(layer_dir, "counterfact_summary.jsonl")

                # numpy conversions
                LO_negmax_arr = np.asarray(LO_negmax_list_by_layer[layer_key], dtype=float)
                all_fail_arr = np.asarray(all_fail_flags_by_layer[layer_key], dtype=int)
                LOc_arr = np.asarray(LOc_list_by_layer[layer_key], dtype=float)
                DISTc_arr = np.asarray(DISTc_list_by_layer[layer_key], dtype=float)
                all_viols_arr = np.asarray(all_viols_by_layer[layer_key], dtype=float)
                intrasims_dict_arr = np.asarray(intrasims_dict[layer_key], dtype=float).mean()
                # failure stats
                total_samples_layer = len(all_fail_arr)
                total_failures_layer = int(all_fail_arr.sum()) if total_samples_layer > 0 else 0
                failure_rate_layer = (total_failures_layer / total_samples_layer) if total_samples_layer > 0 else 0.0

                margin_violation_layer = float(np.nansum(all_viols_arr)) if all_viols_arr.size > 0 else 0.0
                avg_margin_violation_layer = (margin_violation_layer / total_failures_layer) if total_failures_layer > 0 else 0.0

                # stratified fail rates by overlap (Q1 vs Q4)
                fail_rate_high = np.nan
                fail_rate_low = np.nan
                fail_rate_gap = np.nan
                rel_risk = np.nan
                if LO_negmax_arr.size > 0 and all_fail_arr.size > 0:
                    q1, q4 = np.quantile(LO_negmax_arr, [0.25, 0.75])
                    low_bin = (LO_negmax_arr <= q1)
                    high_bin = (LO_negmax_arr >= q4)
                    if low_bin.any():
                        fail_rate_low = float(all_fail_arr[low_bin].mean())
                    if high_bin.any():
                        fail_rate_high = float(all_fail_arr[high_bin].mean())
                    if low_bin.any() and high_bin.any():
                        fail_rate_gap = float(fail_rate_high - fail_rate_low)
                        rel_risk = float(fail_rate_high / max(fail_rate_low, 1e-12))

                # predictive power (AUC)
                auc_overlap = np.nan
                if all_fail_arr.size > 0 and (all_fail_arr.min() != all_fail_arr.max()):
                    try:
                        auc_overlap = float(roc_auc_score(all_fail_arr, LOc_arr))
                    except Exception:
                        auc_overlap = np.nan

                # failure severity gap (within failures)
                sev_gap = np.nan
                mask_fail = (all_fail_arr == 1)
                if mask_fail.any():
                    Jf = LO_negmax_arr[mask_fail]
                    Vf = all_viols_arr[mask_fail]
                    if Jf.size > 0:
                        jf_q1, jf_q4 = np.quantile(Jf, [0.25, 0.75])
                        lo_f = Vf[Jf <= jf_q1]
                        hi_f = Vf[Jf >= jf_q4]
                        if lo_f.size > 0 and hi_f.size > 0:
                            sev_gap = float(np.median(hi_f) - np.median(lo_f))

                with open(summary_path, 'a') as jsonl_file_writer:
                    json.dump({
                        "layer": layer_key,
                        "intrasims_dict": float(intrasims_dict_arr),
                        "Failure Rate": failure_rate_layer,
                        "Average Margin Violation": avg_margin_violation_layer,
                        "FailRate_high_Q4": None if np.isnan(fail_rate_high) else fail_rate_high,
                        "FailRate_low_Q1":  None if np.isnan(fail_rate_low)  else fail_rate_low,
                        "FailRate_gap_Q4_minus_Q1": None if np.isnan(fail_rate_gap) else fail_rate_gap,
                        "RelativeRisk_high_over_low": None if np.isnan(rel_risk) else rel_risk,
                        "AUC_LOcontrast_to_failure": None if np.isnan(auc_overlap) else auc_overlap,
                        "Failure_severity_median_gap_Q4_minus_Q1": None if np.isnan(sev_gap) else sev_gap
                    }, jsonl_file_writer)
                    jsonl_file_writer.write("\n")

                # optional: per-layer plots
                _ = analyze_and_save_distances(
                    incorrect_pairs_by_layer[layer_key],
                    correct_pairs_by_layer[layer_key],
                    title_prefix=f"Counterfact_Layer_{layer_key}",
                    out_dir=layer_dir,
                    group_neg_name="Non-Paraphrase",
                    group_pos_name="Paraphrase",
                    neg_text_pairs=neg_pairs_by_layer[layer_key],
                    pos_text_pairs=pos_pairs_by_layer[layer_key],
                    neg_color="#2EA884",
                    pos_color="#D7732D",
                    hist_alpha=0.35,
                    kde_linewidth=2.0,
                    ecdf_linewidth=2.0,
                    tau_color="#000000",
                    tau_linestyle="--",
                    tau_linewidth=1.5,
                    violin_facealpha=0.35,
                    box_facealpha=0.35,
                )

               


                
                


            
                
                

  




def gemma_test_direct_counterfact_easyedit(file_path,model,file_save_path,tokenizer,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1
    for name, module in model.named_modules():
        print(name, ":", module.__class__.__name__)

 

    # Read the JSON file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # If it's a list of dictionaries, convert to DataFrame
    df = pd.DataFrame(data)


    # Example: access specific fields
    total_samples=0
    fails=0
    l=["model.layers.27"]
    with open(file_save_path, 'w') as jsonl_file_writer:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for row in tqdm(data, desc="Processing rows"):
                    inputs=tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    caption_tensor=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]

                    inputs=tokenizer(row["rephrase_prompt"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    caption2_tensor=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]


                    inputs=tokenizer(row["locality_prompt"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    negative_caption_tensor=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]

                    # caption_tensor = model.encode([row["prompt"]], convert_to_tensor=True).squeeze(0)
                    # caption2_tensor = model.encode([row["rephrase_prompt"]], convert_to_tensor=True).squeeze(0)
                    # negative_caption_tensor = model.encode([row["locality_prompt"]], convert_to_tensor=True).squeeze(0)
                    caption_tensor = torch.nn.functional.normalize(caption_tensor, p=2, dim=0)
                    caption2_tensor = torch.nn.functional.normalize(caption2_tensor, p=2, dim=0)
                    negative_caption_tensor = torch.nn.functional.normalize(negative_caption_tensor, p=2, dim=0)
                                    # --- Cosine similarities ---
                    cos_cap1_cap2   = F.cosine_similarity(caption_tensor, caption2_tensor, dim=0).item()
                    cos_cap1_neg    = F.cosine_similarity(caption_tensor, negative_caption_tensor, dim=0).item()
                    cos_cap2_neg    = F.cosine_similarity(caption2_tensor, negative_caption_tensor, dim=0).item()

                    # --- Euclidean distances ---
                    dist_cap1_cap2  = torch.norm(caption_tensor - caption2_tensor, p=2).item()
                    dist_cap1_neg   = torch.norm(caption_tensor - negative_caption_tensor, p=2).item()
                    dist_cap2_neg   = torch.norm(caption2_tensor - negative_caption_tensor, p=2).item()


                    json_item={"distance_failure": ((dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2)),
                                                "similarity_failure": ((cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2)),
                                                "distances":{"dist_cap1_cap2":dist_cap1_cap2,"dist_cap1_neg":dist_cap1_neg,"dist_cap2_neg":dist_cap2_neg},
                                                "similarities":{"cos_cap1_cap2":cos_cap1_cap2,"cos_cap1_neg":cos_cap1_neg,"cos_cap2_neg":cos_cap2_neg},
                                                "caption":row["prompt"],"caption2":row["rephrase_prompt"],"negative_caption":row["locality_prompt"]}

                    if (dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2) or (cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2):
                        fails+=1
                
                        json.dump(json_item, jsonl_file_writer)
                        jsonl_file_writer.write("\n")

                    total_samples+=1

            print("Accuracy:", (total_samples - fails) / total_samples if total_samples > 0 else 0)
            json.dump({"Accuracy": (total_samples - fails) / total_samples if total_samples > 0 else 0  }, jsonl_file_writer)
            jsonl_file_writer.write("\n")
                

def gemma_embeddings_analysis_counterfact_lasttoken(file_path,model,tokenizer,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1

    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    l=["model.layers.27"]
    with open(file_save_path, 'w') as jsonl_file_writer:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
                for i in tqdm(range(500), desc="Processing 500 steps"):
                    data_dict={}
                    data_entry = json.loads(linecache.getline(file_path, counter).strip())
                    # print(data_entry.keys())
                    torch.cuda.empty_cache()
                    # print(data_entry["edited_prompt"])
                    data_entry["vector_edited_prompt"]=[]
                    inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    data_dict["edit_tensor"]=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]
                    
                    # print([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0].shape)
                    
                    # data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    torch.cuda.empty_cache()
                    # data_entry["vector_edited_prompt_paraphrases_processed"]=[]
                    data_dict["paraphrases_vectors"]=[]
                    paraphrase_strings=[]
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
                    paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed"])
                    # data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    
                    # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
                    torch.cuda.empty_cache()
                    
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
                    paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed_testing"])

                    data_dict["locality_vectors"]=[]
                    locality_strings=[]
                    # data_entry["vectors_neighborhood_prompts_high_sim"]=[]
                    for string in data_entry["neighborhood_prompts_high_sim"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, output_hidden_states=True)
                        data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
                        locality_strings.append(string)
                        # data_entry["vectors_neighborhood_prompts_high_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                    # data_entry["vectors_neighborhood_prompts_low_sim"]=[]
                    
                    for string in data_entry["neighborhood_prompts_low_sim"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, output_hidden_states=True)
                        data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
                        locality_strings.append(string)
                        # data_entry["vectors_neighborhood_prompts_low_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                    for string in data_entry["openai_usable_paraphrases"]:
                        torch.cuda.empty_cache()
                        paraphrase_strings.append(string)
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
                        outputs = model(inputs, output_hidden_states=True)
                        data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
                    # if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
                    #     data_entry["openai_usable_paraphrases_embeddings"]=[]
                    # data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])
                


                    # Get main embedding
                    edit_vec = data_dict["edit_tensor"]  # shape: [3072]

                    # Stack vectors
                    paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # shape: [N_p, 3072]
                    locality_vecs = torch.stack(data_dict["locality_vectors"])        # shape: [N_l, 3072]
                    
                    edit_vec = F.normalize(edit_vec, dim=0)
                    paraphrase_vecs = F.normalize(paraphrase_vecs, dim=1)
                    locality_vecs = F.normalize(locality_vecs, dim=1)
                    # 1. Distances from edit → each paraphrase
                    para_sims = F.cosine_similarity(
                        paraphrase_vecs, edit_vec.unsqueeze(0), dim=1
                    )  # shape: [N_p], similarity scores

                    # 2. Distances from edit → each locality vector
                    local_sims = F.cosine_similarity(
                        locality_vecs, edit_vec.unsqueeze(0), dim=1
                    )  # shape: [N_l], similarity scores

                    # # 3. Compare
                    # max_para_sim = para_distances.max().item()  # highest similarity
                    # max_local_sim = local_distances.max().item()

                    # print(f"Closest paraphrase sim: {max_para_sim:.4f}")
                    # print(f"Closest locality sim:  {max_local_sim:.4f}")
                    # counter+=1
                        # json.dump(data_entry, jsonl_file_writer)
                        # jsonl_file_writer.write('\n')

                    # Ensure everything is on the same device
                    edit_vec = data_dict["edit_tensor"]  # shape: [3072]
                    paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # [N_p, 3072]
                    locality_vecs = torch.stack(data_dict["locality_vectors"])        # [N_l, 3072]

                    edit_vec = edit_vec.to(paraphrase_vecs.device)

                    # Compute Euclidean distances
                    para_dists = torch.norm(paraphrase_vecs - edit_vec.unsqueeze(0), dim=1)  # [N_p]
                    local_dists = torch.norm(locality_vecs - edit_vec.unsqueeze(0), dim=1)   # [N_l]

                    # Min distances
                    min_para_dist = para_dists.min().item()
                    min_local_dist = local_dists.min().item()

                    # Print for debug
                    # print(f"Closest paraphrase distance: {min_para_dist:.4f}")
                    # print(f"Closest locality distance:   {min_local_dist:.4f}")
                    violating_pairs=[]
                    for i, (local_dist,local_sim) in enumerate(zip(local_dists,local_sims)):
                        for j, (para_dist, para_sim ) in enumerate(zip(para_dists,para_sims)):
                            if local_dist < para_dist or local_sim > para_sim:
                                # print("heelo")
                                violating_pairs.append({"distance_failure": (local_dist < para_dist).item(),"similarity_failure": (local_sim > para_sim).item(),"distances_sims":{"local_dist":local_dist.item(),"para_dist":para_dist.item(),"local_sim":local_sim.item(),"para_sim":para_sim.item()},"edit":data_entry["edited_prompt"][0],"paraphrase":paraphrase_strings[j],"locality":locality_strings[i]})
                    
                    
                    for entry in violating_pairs:
                        # print(entry)
                        json.dump(entry, jsonl_file_writer)
                        jsonl_file_writer.write("\n")
                    # if violating_pairs:       
                    #     print(violating_pairs)     
                    # Check comparison
                    # if min_local_dist < min_para_dist or max_local_sim > max_para_sim:
                    #     print("⚠️ A locality vector is closer to the edit than any paraphrase (Euclidean).",data_entry["edited_prompt"][0],paraphrase_strings[j],locality_strings[i])
                    counter+=1
                # break


def gemma_embeddings_analysis_counterfact_average(file_path,model,tokenizer,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1

    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    l=["model.layers.27"]
    with open(file_save_path, 'w') as jsonl_file_writer:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
                for i in tqdm(range(500), desc="Processing 500 steps"):
                    data_dict={}
                    data_entry = json.loads(linecache.getline(file_path, counter).strip())
                    # print(data_entry.keys())
                    torch.cuda.empty_cache()
                    # print(data_entry["edited_prompt"])
                    data_entry["vector_edited_prompt"]=[]
                    inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    data_dict["edit_tensor"]=[ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0]
                   
                    # print([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0].shape)
                    
                    # data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    torch.cuda.empty_cache()
                    # data_entry["vector_edited_prompt_paraphrases_processed"]=[]
                    data_dict["paraphrases_vectors"]=[]
                    paraphrase_strings=[]
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
                    paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed"])
                    # data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    
                    # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
                    torch.cuda.empty_cache()
                    
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
                    paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed_testing"])

                    data_dict["locality_vectors"]=[]
                    locality_strings=[]
                    # data_entry["vectors_neighborhood_prompts_high_sim"]=[]
                    for string in data_entry["neighborhood_prompts_high_sim"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, output_hidden_states=True)
                        data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
                        locality_strings.append(string)
                        # data_entry["vectors_neighborhood_prompts_high_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                    # data_entry["vectors_neighborhood_prompts_low_sim"]=[]
                    
                    for string in data_entry["neighborhood_prompts_low_sim"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, output_hidden_states=True)
                        data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
                        locality_strings.append(string)
                        # data_entry["vectors_neighborhood_prompts_low_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                    for string in data_entry["openai_usable_paraphrases"]:
                        torch.cuda.empty_cache()
                        paraphrase_strings.append(string)
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
                        outputs = model(inputs, output_hidden_states=True)
                        data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
                    # if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
                    #     data_entry["openai_usable_paraphrases_embeddings"]=[]
                    # data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])
            

                    # Get main embedding
                    edit_vec = data_dict["edit_tensor"]  # shape: [3072]

                    # Stack vectors
                    paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # shape: [N_p, 3072]
                    locality_vecs = torch.stack(data_dict["locality_vectors"])        # shape: [N_l, 3072]
                    
                    edit_vec = F.normalize(edit_vec, dim=0)
                    paraphrase_vecs = F.normalize(paraphrase_vecs, dim=1)
                    locality_vecs = F.normalize(locality_vecs, dim=1)
                    # 1. Distances from edit → each paraphrase
                    para_sims = F.cosine_similarity(
                        paraphrase_vecs, edit_vec.unsqueeze(0), dim=1
                    )  # shape: [N_p], similarity scores

                    # 2. Distances from edit → each locality vector
                    local_sims = F.cosine_similarity(
                        locality_vecs, edit_vec.unsqueeze(0), dim=1
                    )  # shape: [N_l], similarity scores

                    # # 3. Compare
                    # max_para_sim = para_distances.max().item()  # highest similarity
                    # max_local_sim = local_distances.max().item()

                    # print(f"Closest paraphrase sim: {max_para_sim:.4f}")
                    # print(f"Closest locality sim:  {max_local_sim:.4f}")
                    # counter+=1
                        # json.dump(data_entry, jsonl_file_writer)
                        # jsonl_file_writer.write('\n')

                    # Ensure everything is on the same device
                    edit_vec = data_dict["edit_tensor"]  # shape: [3072]
                    paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # [N_p, 3072]
                    locality_vecs = torch.stack(data_dict["locality_vectors"])        # [N_l, 3072]

                    edit_vec = edit_vec.to(paraphrase_vecs.device)

                    # Compute Euclidean distances
                    para_dists = torch.norm(paraphrase_vecs - edit_vec.unsqueeze(0), dim=1)  # [N_p]
                    local_dists = torch.norm(locality_vecs - edit_vec.unsqueeze(0), dim=1)   # [N_l]

                    # Min distances
                    min_para_dist = para_dists.min().item()
                    min_local_dist = local_dists.min().item()

                    # Print for debug
                    # print(f"Closest paraphrase distance: {min_para_dist:.4f}")
                    # print(f"Closest locality distance:   {min_local_dist:.4f}")
                    violating_pairs=[]
                    for i, (local_dist,local_sim) in enumerate(zip(local_dists,local_sims)):
                        for j, (para_dist, para_sim ) in enumerate(zip(para_dists,para_sims)):
                            if local_dist < para_dist or local_sim > para_sim:
                                # print("heelo")
                                violating_pairs.append({"distance_failure": (local_dist < para_dist).item(),"similarity_failure": (local_sim > para_sim).item(),"distances_sims":{"local_dist":local_dist.item(),"para_dist":para_dist.item(),"local_sim":local_sim.item(),"para_sim":para_sim.item()},"edit":data_entry["edited_prompt"][0],"paraphrase":paraphrase_strings[j],"locality":locality_strings[i]})
                    
                    
                    for entry in violating_pairs:
                        json.dump(entry, jsonl_file_writer)
                        jsonl_file_writer.write("\n")
                    # if violating_pairs:       
                    #     print(violating_pairs)     
                    # Check comparison
                    # if min_local_dist < min_para_dist or max_local_sim > max_para_sim:
                    #     print("⚠️ A locality vector is closer to the edit than any paraphrase (Euclidean).",data_entry["edited_prompt"][0],paraphrase_strings[j],locality_strings[i])
                    counter+=1
                # break