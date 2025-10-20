
import torch
import nethook
import json,os
from tqdm import tqdm
import linecache
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from visualization_quora_paws import analyze_and_save_distances
import numpy as np
from helper_functions import _counts, _jaccard_overlap_pct,matrix_entropy_batch_from_tokens
import random
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from collections import defaultdict
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
def load_model(model_name="llama",access_token=None,device="auto"):
    model_name = "meta-llama/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 if needed
        device_map=device,
        token=access_token,  # Use access token if required
    )
    model.eval()
    return tokenizer, model



#----------------------------------------------------------------------------
# Section: Llama model Instruct Analysis Counterfact
#----------------------------------------------------------------------------
#region Llama model Instruct Analysis Counterfact
def build_batched_llama_chat_inputs(tokenizer, texts, add_generation_prompt=True, device="auto"):
    """
    Build a batch of chat-formatted prompts for Llama-3.2-* Instruct using tokenizer.apply_chat_template.
    `texts` is a list[str].
    Returns: dict with input_ids, attention_mask on device.
    """
    conversations = [
        [
            {"role": "user", "content": [{"type": "text", "text": t.lower()}]}
        ]
        for t in texts
    ]
    enc = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def get_mask_text_llama(inputs, tokenizer, texts, embeddings, *,
                          lowercase_texts=True, device="auto"):

    if device == "auto":
        device = embeddings.device

    # Normalize texts the same way you did in build_batched_llama_chat_inputs
    if lowercase_texts:
        norm_texts = [t.lower() for t in texts]
    else:
        norm_texts = list(texts)

    # Tokenize raw texts WITHOUT special/chat tokens to get the pure content pattern
    # (important: add_special_tokens=False)
    tok = tokenizer(norm_texts, add_special_tokens=False)

    # Accept either (B, L, H) or (T, B, L, H)
    if embeddings.ndim == 3:
        B, L, _H = embeddings.shape
        T = None
    elif embeddings.ndim == 4:
        T, B, L, _H = embeddings.shape
    else:
        raise ValueError(f"embeddings must be 3D or 4D, got {tuple(embeddings.shape)}")

    input_ids_batch = inputs["input_ids"]        # (B, L)
    attention_mask  = inputs["attention_mask"]   # (B, L)

    base_mask = torch.zeros((B, L, 1), dtype=embeddings.dtype, device=device)

    # For each item in the batch, reverse-search the pure-text token pattern
    for i, pat in enumerate(tok["input_ids"]):
        seq = input_ids_batch[i].tolist()
        m = len(pat)
        best_j = -1

        if m > 0 and m <= len(seq):
            # reverse search: prefer the last occurrence (most recent message)
            for j in range(len(seq) - m, -1, -1):
                if seq[j:j+m] == pat:
                    best_j = j
                    break

        if best_j >= 0:
            base_mask[i, best_j:best_j+m, 0] = 1.0
        else:
            # Fallback: use attention_mask so the row isn't empty
            base_mask[i] = attention_mask[i].unsqueeze(-1).type_as(base_mask)

    # Safety: guarantee at least 1 token selected per row
    empty = (base_mask.sum(dim=1) == 0).squeeze(-1)
    if empty.any():
        base_mask[empty] = attention_mask[empty].unsqueeze(-1).type_as(base_mask)

    # Broadcast across layers if needed
    if T is not None:
        return base_mask.unsqueeze(0).expand(T, B, L, 1)
    return base_mask

def to_1d_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x).reshape(-1)
    return x
def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)

def summarize(arr, mask, name):
    m = np.mean(arr[mask]) if mask.any() else np.nan
    s = np.std(arr[mask]) if mask.any() else np.nan
    q = np.percentile(arr[mask], [25,50,75]) if mask.any() else [np.nan]*3
    print(f"{name}: mean={m:.4f} std={s:.4f} q25/50/75={q}")

def select_topn_dims_by_variance(emb, mask,percentage_selection):
    """
    emb:  (B, T, H)  - full embeddings
    mask: (B, T, 1)  - 1 for valid tokens, 0 for padding (broadcastable over H)
    n:    int        - number of feature dimensions to keep (per batch element)

    Returns:
      emb_reduced: (B, T, n)  - embeddings restricted to top-n dims per sample
      topn_idx:    (B, n)     - indices of selected dims in original H
      var:         (B, H)     - variance used for selection
    """
    B, T, H = emb.shape
    # Zero-out pads
    # emb_z = emb * mask  # (B, T, H)
    # print("mask",mask.shape,"embz",emb.shape)
    # Per-sequence mean across time (ignore pads)
    lengths = mask.sum(dim=1).clamp(min=1)              # (B, 1, 1)
    mean = emb.sum(dim=1) / lengths     # (B, 1, H)
    # print("--------------------\nmean",mean.shape,lengths.shape)
    # Per-sequence variance across time (ignore pads)
    sq_diff = ((emb - mean.unsqueeze(1)) ** 2) * mask           # (B, T, H)
    # print("sq_diff",sq_diff.shape)
    var = sq_diff.sum(dim=1) / lengths     # (B, H)
    k=int(H*percentage_selection)
    # Top-n dims per batch element
    topn_idx = var.topk(k, dim=1, largest=True).indices               # (B, n)

    # Gather those dims for every timestep in that sample
    idx_expanded = topn_idx.unsqueeze(1).expand(-1, T, -1)  # (B, T, n)

    emb_reduced = torch.gather(emb, dim=2, index=idx_expanded)  # (B, T, n)
    # print("emb_reduced",emb_reduced.shape)
    return emb_reduced, topn_idx, var

def llama_it_counterfact_scpp(data_loader,args,acess_token,layers,device):
    """
    Run PAWS-like pair distance extraction on Llama 3.2 Instruct checkpoints.
    Expects:
      - args.model_type in {"Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"} (without 'meta-llama/' prefix)
      - load_model(), acess_token_gemma, nethook, etc. defined in your env (mirrors your Gemma code)
      - args.save_path exists
    """
    import os, json, torch
    from tqdm import tqdm



    print("Loading model...")
    tokenizer, model = load_model(args.model_type, acess_token, device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    
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

    percentage_selection=0.30
     # for scatter plot
    l = layers
    
    with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
    
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
                # Sentence 1
                inputs = build_batched_llama_chat_inputs(tokenizer, anchors, add_generation_prompt=True, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                anchor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                mask = get_mask_text_llama(inputs,tokenizer,anchors,anchor_sentence_embeddings,device=device)
                if(args.mode):
                    # print([ret[layer_key].output.shape for layer_key in ret])
                   
                    # print("mask",mask.shape)
                    # print(mask[0][0])
                    # inputs["attention_mask"].unsqueeze(-1)
                    padded_zero_embeddings=anchor_sentence_embeddings * mask
                    # print("padded_zero_embeddings",padded_zero_embeddings.shape)
                    # padded_zero_embeddings,_,_=select_topn_dims_by_variance(padded_zero_embeddings, mask,percentage_selection)

                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, H)
                 
                    lengths = mask.sum(dim=2)           # (B, 1)
                   
                    anchor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    # return None
                else: 
                    L, B, T, E = anchor_sentence_embeddings.shape
                    base_mask = mask[0].squeeze(-1).to(anchor_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=anchor_sentence_embeddings.device, dtype=anchor_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    anchor_sentence_embeddings = anchor_sentence_embeddings.gather(2, idx).squeeze(2)
                    # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                    # return None
                    # mask = inputs["input_ids"] != tokenizer.pad_token_id
                    # last_index = mask.sum(dim=1) - 1
                    # anchor_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]
                
                # Sentence 2
                inputs = build_batched_llama_chat_inputs(tokenizer, paraphrases, add_generation_prompt=True, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret],dim=0)    
                mask = get_mask_text_llama(inputs,tokenizer,paraphrases,paraphrase_sentence_embeddings,device=device)
                if(args.mode):
                    
                    padded_zero_embeddings=paraphrase_sentence_embeddings * mask
                    # padded_zero_embeddings,_,_=select_topn_dims_by_variance(padded_zero_embeddings, mask,percentage_selection)
                    # paraphrase_entropies=matrix_entropy_batch_from_tokens(padded_zero_embeddings)
                    # zero out pads, then mean over valid tokens
                    # masked_embeddings = padded_zero_embeddings.masked_fill(mask == 0, -1e9)
                    # paraphrase_sentence_embeddings, _ = masked_embeddings.max(dim=1)
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, H)
                    lengths = mask.sum(dim=2)                                     # (B, 1)
                    paraphrase_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    
                else:
                    L, B, T, E = paraphrase_sentence_embeddings.shape
                    base_mask = mask[0].squeeze(-1).to(paraphrase_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=paraphrase_sentence_embeddings.device, dtype=paraphrase_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    paraphrase_sentence_embeddings = paraphrase_sentence_embeddings.gather(2, idx).squeeze(2)
                    # mask = inputs["input_ids"] != tokenizer.pad_token_id
                    # last_index = mask.sum(dim=1) - 1
                    # paraphrase_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]
                # Sentence 3
                inputs = build_batched_llama_chat_inputs(tokenizer, distractors, add_generation_prompt=True, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                distractor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                mask = get_mask_text_llama(inputs,tokenizer,distractors,distractor_sentence_embeddings,device=device) #(B,L,T)
                if(args.mode):
                    
                    padded_zero_embeddings=distractor_sentence_embeddings * mask
                    # padded_zero_embeddings,_,_=select_topn_dims_by_variance(padded_zero_embeddings, mask,percentage_selection)
                    # distractor_entropies=matrix_entropy_batch_from_tokens(padded_zero_embeddings)
                    # masked_embeddings = padded_zero_embeddings.masked_fill(mask == 0, -1e9)
                    # distractor_sentence_embeddings, _ = masked_embeddings.max(dim=1)
                    # zero out pads, then mean over valid tokens
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
                    lengths = mask.sum(dim=2)                                     # (B, ,L, 1)
                    distractor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    
                else:
                    L, B, T, E = distractor_sentence_embeddings.shape
                    base_mask = mask[0].squeeze(-1).to(distractor_sentence_embeddings.dtype)
                    token_range = torch.arange(T, device=distractor_sentence_embeddings.device, dtype=distractor_sentence_embeddings.dtype)
                    last_idx = (base_mask * token_range).argmax(dim=1).long()
                    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
                    distractor_sentence_embeddings = distractor_sentence_embeddings.gather(2, idx).squeeze(2)

                    # mask = inputs["input_ids"] != tokenizer.pad_token_id
                    # last_index = mask.sum(dim=1) - 1
                    # distractor_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

                

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
                    
                    # def cos64(a, b, clamp_eps=1e-7):
                    #     a64 = a.to(torch.float64)
                    #     b64 = b.to(torch.float64)
                    #     c = (a64 * b64).sum(dim=1)
                    #     c = torch.clamp(c, -1.0 + clamp_eps, 1.0 - clamp_eps)
                    #     return c

                    # cos_ap = cos64(anchor_sentence_embeddings, paraphrase_sentence_embeddings)   # [32]
                    # cos_ad = cos64(anchor_sentence_embeddings, distractor_sentence_embeddings)
                    # cos_pd = cos64(paraphrase_sentence_embeddings, distractor_sentence_embeddings)

                    # # 3) Arc (geodesic) distances (radians); back to float32 if you like
                    # distances_anchor_paraphrase = torch.arccos(cos_ap).to(torch.float32)
                    # distances_anchor_distractor = torch.arccos(cos_ad).to(torch.float32)
                    # distances_paraphrase_distractor = torch.arccos(cos_pd).to(torch.float32)
                # print("distances_anchor_paraphrase",distances_anchor_paraphrase.shape)
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
# region Llama model Pretrained Analysis
def build_batched_llama_pt_inputs(tokenizer, texts, device="auto"):
    """
    Build a batch for Llama-3.2 base (non-instruct) using plain tokenization.
    """
    enc = tokenizer(
        [t.lower() for t in texts],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def llama_pt_counterfact_scpp(data_loader,args,access_token,layers,device="auto"):
    tokenizer,model = load_model(args.model_type,access_token,device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    percentage_selection=0.30
     # for scatter plot
    l = layers
    
    with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
    
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
                # Sentence 1
                inputs = build_batched_llama_pt_inputs(tokenizer, anchors, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                if(args.mode):
                    # print([ret[layer_key].output.shape for layer_key in ret])
                    anchor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                    # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                    # attention mask (B, L) -> (B, L, 1)
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    padded_zero_embeddings=anchor_sentence_embeddings * mask
                    # padded_zero_embeddings,_,_=select_topn_dims_by_variance(padded_zero_embeddings, mask,percentage_selection)

                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, H)
                    lengths = mask.sum(dim=1)           # (B, 1)
                    anchor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)

                else:
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1                                     # (B,)

                    distractor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret], dim=0)  # (L, B, T, H)
                    idx = last_index.view(1, -1, 1, 1).expand(distractor_sentence_embeddings.size(0),
                                                            distractor_sentence_embeddings.size(1),
                                                            1,
                                                            distractor_sentence_embeddings.size(3))                  # (L, B, 1, H)
                    anchor_sentence_embeddings = torch.gather(distractor_sentence_embeddings, dim=2, index=idx).squeeze(2)  # (L, B, H)

                    # mask = inputs["input_ids"] != tokenizer.pad_token_id
                    # last_index = mask.sum(dim=1) - 1
                    # anchor_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]
                
                # Sentence 2
                inputs = build_batched_llama_pt_inputs(tokenizer, paraphrases, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                if(args.mode):
                    paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                    # attention mask (B, L) -> (B, L, 1)
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    padded_zero_embeddings=paraphrase_sentence_embeddings * mask
                    # padded_zero_embeddings,_,_=select_topn_dims_by_variance(padded_zero_embeddings, mask,percentage_selection)
                    # paraphrase_entropies=matrix_entropy_batch_from_tokens(padded_zero_embeddings)
                    # zero out pads, then mean over valid tokens
                    # masked_embeddings = padded_zero_embeddings.masked_fill(mask == 0, -1e9)
                    # paraphrase_sentence_embeddings, _ = masked_embeddings.max(dim=1)
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, H)
                    lengths = mask.sum(dim=1)                                     # (B, 1)
                    paraphrase_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    
                else:
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1                                     # (B,)

                    distractor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret], dim=0)  # (L, B, T, H)
                    idx = last_index.view(1, -1, 1, 1).expand(distractor_sentence_embeddings.size(0),
                                                            distractor_sentence_embeddings.size(1),
                                                            1,
                                                            distractor_sentence_embeddings.size(3))                  # (L, B, 1, H)
                    paraphrase_sentence_embeddings = torch.gather(distractor_sentence_embeddings, dim=2, index=idx).squeeze(2)  # (L, B, H)
                    # mask = inputs["input_ids"] != tokenizer.pad_token_id
                    # last_index = mask.sum(dim=1) - 1
                    # paraphrase_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]
                # Sentence 3
                inputs = build_batched_llama_pt_inputs(tokenizer, distractors, device=device)
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                if(args.mode):
                    distractor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                    # attention mask (B, L) -> (B, L, 1)
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    padded_zero_embeddings=distractor_sentence_embeddings * mask
                    # padded_zero_embeddings,_,_=select_topn_dims_by_variance(padded_zero_embeddings, mask,percentage_selection)
                    # distractor_entropies=matrix_entropy_batch_from_tokens(padded_zero_embeddings)
                    # masked_embeddings = padded_zero_embeddings.masked_fill(mask == 0, -1e9)
                    # distractor_sentence_embeddings, _ = masked_embeddings.max(dim=1)
                    # zero out pads, then mean over valid tokens
                    sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, H)
                    lengths = mask.sum(dim=1)                                     # (B, 1)
                    distractor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                    
                else:
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1                                     # (B,)

                    distractor_sentence_embeddings = torch.stack([ret[layer_key].output for layer_key in ret], dim=0)  # (L, B, T, H)
                    idx = last_index.view(1, -1, 1, 1).expand(distractor_sentence_embeddings.size(0),
                                                            distractor_sentence_embeddings.size(1),
                                                            1,
                                                            distractor_sentence_embeddings.size(3))                  # (L, B, 1, H)
                    distractor_sentence_embeddings = torch.gather(distractor_sentence_embeddings, dim=2, index=idx).squeeze(2)  # (L, B, H)
                    
                    # mask = inputs["input_ids"] != tokenizer.pad_token_id
                    # last_index = mask.sum(dim=1) - 1
                    # distractor_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]


            
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
                
                    # def cos64(a, b, clamp_eps=1e-7):
                    #     a64 = a.to(torch.float64)
                    #     b64 = b.to(torch.float64)
                    #     c = (a64 * b64).sum(dim=1)
                    #     c = torch.clamp(c, -1.0 + clamp_eps, 1.0 - clamp_eps)
                    #     return c

                    # cos_ap = cos64(anchor_sentence_embeddings, paraphrase_sentence_embeddings)   # [32]
                    # cos_ad = cos64(anchor_sentence_embeddings, distractor_sentence_embeddings)
                    # cos_pd = cos64(paraphrase_sentence_embeddings, distractor_sentence_embeddings)

                    # # 3) Arc (geodesic) distances (radians); back to float32 if you like
                    # distances_anchor_paraphrase = torch.arccos(cos_ap).to(torch.float32)
                    # distances_anchor_distractor = torch.arccos(cos_ad).to(torch.float32)
                    # distances_paraphrase_distractor = torch.arccos(cos_pd).to(torch.float32)
                # print("distances_anchor_paraphrase",distances_anchor_paraphrase.shape)
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
    
# def llama_pt_counterfact_scpp(data_loader,args,acess_token,layer_mapping_dict,device):
#     """
#     Run PAWS-like pair distance extraction on Llama 3.2 Instruct checkpoints.
#     Expects:
#       - args.model_type in {"Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"} (without 'meta-llama/' prefix)
#       - load_model(), acess_token_gemma, nethook, etc. defined in your env (mirrors your Gemma code)
#       - args.save_path exists
#     """



#     print("Loading model...")
#     tokenizer, model = load_model(args.model_type, acess_token, device)
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.eos_token_id
#     model.eval()

#     incorrect_pairs=[]
#     correct_pairs=[]
#     neg_pairs=[]
#     pos_pairs=[]
#     average_margin_lor_low=0
#     average_margin_violation_lor_low=0
#     failure_rate_lor_low=0
#     average_margin_lor_high=0
#     average_margin_violation_lor_high=0
#     failure_rate_lor_high=0
#     l = layer_mapping_dict[args.model_type]
    

#     file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with nethook.TraceDict(model, l) as ret:
#             with torch.no_grad():
#                 for batch in tqdm(data_loader, desc="Processing Rows"):
        
#                     anchors, paraphrases, distractors, LOS_flag = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
#                     # Sentence 1
#                     inputs = build_batched_llama_pt_inputs(tokenizer, anchors, device=device)
#                     mask = inputs["input_ids"] != tokenizer.pad_token_id
#                     last_index = mask.sum(dim=1) - 1
#                     _ = model(
#                         input_ids=inputs["input_ids"],
#                         attention_mask=inputs["attention_mask"],
#                         output_hidden_states=True
#                     )
#                     anchor_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

#                     # Sentence 2
#                     inputs = build_batched_llama_pt_inputs(tokenizer, paraphrases, device=device)
#                     mask = inputs["input_ids"] != tokenizer.pad_token_id
#                     last_index = mask.sum(dim=1) - 1
#                     _ = model(
#                         input_ids=inputs["input_ids"],
#                         attention_mask=inputs["attention_mask"],
#                         output_hidden_states=True
#                     )
#                     paraphrase_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]
#                     # Sentence 3
#                     inputs = build_batched_llama_pt_inputs(tokenizer, distractors, device=device)
#                     mask = inputs["input_ids"] != tokenizer.pad_token_id
#                     last_index = mask.sum(dim=1) - 1
#                     _ = model(
#                         input_ids=inputs["input_ids"],
#                         attention_mask=inputs["attention_mask"],
#                         output_hidden_states=True
#                     )
#                     distractor_sentence_embeddings=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

#                     anchor_sentence_embeddings = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=1)
#                     paraphrase_sentence_embeddings = torch.nn.functional.normalize(paraphrase_sentence_embeddings, p=2, dim=1)
#                     distractor_sentence_embeddings = torch.nn.functional.normalize(distractor_sentence_embeddings, p=2, dim=1)
#                     # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
#                     # --- Cosine similarities ---
#                     cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embeddings, paraphrase_sentence_embeddings, dim=1)
#                     cosine_anchor_distractor    = F.cosine_similarity(anchor_sentence_embeddings, distractor_sentence_embeddings, dim=1)
#                     cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_sentence_embeddings, distractor_sentence_embeddings, dim=1)

#                     # --- Euclidean distances ---
#                     distance_anchor_paraphrase = torch.norm(anchor_sentence_embeddings - paraphrase_sentence_embeddings, p=2,dim=1)
#                     distance_anchor_distractor   = torch.norm(anchor_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=1)
#                     distance_paraphrase_distractor = torch.norm(paraphrase_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=1)
#                     # print("distance_anchor_paraphrase",distance_anchor_paraphrase)
#                     # print("distance_anchor_distractor",distance_anchor_distractor)
#                     # print("distance_paraphrase_distractor",distance_paraphrase_distractor)
#                     nearest_distractors = torch.minimum(distance_anchor_distractor, distance_paraphrase_distractor)  
#                     distance_violations   = (nearest_distractors < distance_anchor_paraphrase)  
#                     similarity_violations = (torch.maximum(cosine_anchor_distractor, cosine_paraphrase_distractor) > cosine_anchor_paraphrase)
              
#                     for i in range(len(anchors)):
#                         json_item={"distance_failure": distance_violations[i].item(),
#                                                     "similarity_failure": similarity_violations[i].item(),
#                                                     "distances":{"dist_cap1_cap2":distance_anchor_paraphrase[i].item(),"dist_cap1_neg":distance_anchor_distractor[i].item(),"dist_cap2_neg":distance_paraphrase_distractor[i].item()},
#                                                     "similarities":{"cos_cap1_cap2":cosine_anchor_paraphrase[i].item(),"cos_cap1_neg":cosine_anchor_distractor[i].item(),"cos_cap2_neg":cosine_paraphrase_distractor[i].item()},
#                                                     "anchor":anchors[i],"paraphrase":paraphrases[i],"distractor":distractors[i]}
#                         json.dump(json_item, jsonl_file_writer)
#                         jsonl_file_writer.write("\n")
                      
#                         condition_anchor=distance_anchor_distractor[i] < distance_anchor_paraphrase[i] 
#                         condition_paraphrase=distance_paraphrase_distractor[i] < distance_anchor_paraphrase[i]
#                         possible_margin_violation=(distance_anchor_paraphrase[i] - min(distance_paraphrase_distractor[i],distance_anchor_distractor[i])).item()
#                         # print("possible_margin_violation",possible_margin_violation)
#                         failure=(condition_anchor or condition_paraphrase).item()
#                         if(LOS_flag[i]=="low"):
#                             if(failure):
#                                 average_margin_violation_lor_low+= possible_margin_violation#add the margin violation for failures
#                                 failure_rate_lor_low+=1#increase failure count for low overlap
#                             average_margin_lor_low+= possible_margin_violation#add the margin violation general
#                         elif(LOS_flag[i]=="high"):
#                             if(failure):
#                                 average_margin_violation_lor_high+= possible_margin_violation#add the margin violation for failures
#                                 failure_rate_lor_high+=1#increase failure count for high overlap
#                             average_margin_lor_high+= possible_margin_violation#add the margin violation general
                        
                        
#                         if(distance_anchor_distractor[i]<=distance_paraphrase_distractor[i]):
#                             incorrect_pairs.append(distance_anchor_distractor[i].item())
#                             neg_pairs.append((anchors[i],distractors[i]))
#                         else:
#                             incorrect_pairs.append(distance_paraphrase_distractor[i].item())
#                             neg_pairs.append((paraphrases[i],distractors[i]))

#                         correct_pairs.append(distance_anchor_paraphrase[i].item())
#                         pos_pairs.append((anchors[i],paraphrases[i]))

                    

                   
            
#             total_samples=len(data_loader.dataset)
#             total_failures=failure_rate_lor_high+failure_rate_lor_low
#             print("total_failures",total_failures)

#             margin_violation=average_margin_violation_lor_high+average_margin_violation_lor_low
#             avg_margin_violation=margin_violation/total_failures if total_failures>0 else 0#average margin violation of the failures
#             print("avg_margin_violation",avg_margin_violation)
#             failure_rate=total_failures/total_samples#failure rate overall

#             LOS_sensitivity = (average_margin_lor_high / data_loader.dataset.count_high_flags if getattr(data_loader.dataset, "count_high_flags", 0)>0 else 0) - \
#                               (average_margin_lor_low  / data_loader.dataset.count_low_flags  if getattr(data_loader.dataset, "count_low_flags", 0)>0  else 0)
#             print("LOS_sensitivity", LOS_sensitivity)

#             LOS_sensitivity_fails= average_margin_violation_lor_high/failure_rate_lor_high if failure_rate_lor_high>0 else 0 - \
#                                     average_margin_violation_lor_low/failure_rate_lor_low if failure_rate_lor_low>0 else 0
#             print("LOS_sensitivity Fails", LOS_sensitivity_fails)

#             json.dump({"Failure Rate": failure_rate,"Average Margin Violation": avg_margin_violation,"LOS_sensitivity":LOS_sensitivity,"LOS_sensitivity_fails":LOS_sensitivity_fails}, jsonl_file_writer)
#             jsonl_file_writer.write("\n")


               

#     _= analyze_and_save_distances(
#         incorrect_pairs,
#         correct_pairs,
#         title_prefix="Counterfact",
#         out_dir=args.save_path,
#         # names
#         group_neg_name="Non-Paraphrase",
#         group_pos_name="Paraphrase",
#         # optional ROUGE text pairs
#         neg_text_pairs=neg_pairs,
#         pos_text_pairs=pos_pairs,
#         # >>> NEW: simple styling hooks <<<
#         neg_color="#2EA884",    # blue
#         pos_color="#D7732D",    # red
#         hist_alpha=0.35,
#         kde_linewidth=2.0,
#         ecdf_linewidth=2.0,
#         tau_color="#000000",    # black
#         tau_linestyle="--",
#         tau_linewidth=1.5,
#         violin_facealpha=0.35,
#         box_facealpha=0.35,
#         )
    
#endregion

def llama_embeddings_analysis_counterfact_lasttoken(file_path,model,tokenizer,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1

    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    l=["model.layers.31"]
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


def llama_embeddings_analysis_counterfact_average(file_path,model,tokenizer,file_save_path,device):
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


    # apd = to_1d_numpy(comparison_entropies_apd)
    # ap  = to_1d_numpy(comparison_entropies_ap)
    # flags = to_1d_numpy(all_fail_flags).astype(int)
    # fail_mask = flags == 1
    # succ_mask = flags == 0
    # H_A = to_np(entropies_anchor)      # list/array of anchor entropies
    # H_P = to_np(entropies_paraphrase)  # list/array of paraphrase entropies
    # H_D = to_np(entropies_distractor) 
    # correct_pairs=to_np(correct_pairs)
    # summarize(apd, succ_mask, "APD Success"); summarize(apd, fail_mask, "APD Failure")
    # summarize(ap,  succ_mask, "AP  Success"); summarize(ap,  fail_mask, "AP  Failure")
    # fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # # --- top row: scatter plots ---
    # axes[0,0].scatter(ap[succ_mask], apd[succ_mask], s=25, alpha=0.6, color="tab:blue")
    # axes[0,0].set_title("Success Cases")
    # axes[0,0].set_xlabel("ap")
    # axes[0,0].set_ylabel("apd")
    # axes[0,0].grid(alpha=0.3)

    # axes[0,1].scatter(ap[fail_mask], apd[fail_mask], s=25, alpha=0.6, color="tab:orange")
    # axes[0,1].set_title("Failure Cases")
    # axes[0,1].set_xlabel("ap")
    # axes[0,1].set_ylabel("apd")
    # axes[0,1].grid(alpha=0.3)
    # # axes[0,1].set_xlim(ap[succ_mask].min(), ap[succ_mask].max())

    # # leave [0,2] empty or use for legend / text
    # axes[0,2].axis("off")
    # axes[0,2].text(0.2, 0.5,
    #             "Top: scatter plots of entropy differences\n"
    #             "Bottom: distributions of absolute entropies\n"
    #             "(A, P, D)",
    #             fontsize=11, va='center')

    # # --- bottom row: absolute entropy distributions ---
    # bins = 80
    # axes[1,0].hist([H_A[succ_mask], H_A[fail_mask]], bins=bins, density=True,
    #             label=["Success", "Failure"], color=["tab:blue","tab:orange"], alpha=0.6)
    # axes[1,0].set_title("Entropy of Anchor (H_A)")
    # axes[1,0].set_xlabel("Entropy")
    # axes[1,0].legend()
    # axes[1,0].grid(alpha=0.3)

    # axes[1,1].hist([H_P[succ_mask], H_P[fail_mask]], bins=bins, density=True,
    #             label=["Success", "Failure"], color=["tab:blue","tab:orange"], alpha=0.6)
    # axes[1,1].set_title("Entropy of Paraphrase (H_P)")
    # axes[1,1].set_xlabel("Entropy")
    # axes[1,1].legend()
    # axes[1,1].grid(alpha=0.3)

    # axes[1,2].hist([H_D[succ_mask], H_D[fail_mask]], bins=bins, density=True,
    #             label=["Success", "Failure"], color=["tab:blue","tab:orange"], alpha=0.6)
    # axes[1,2].set_title("Entropy of Distractor (H_D)")
    # axes[1,2].set_xlabel("Entropy")
    # axes[1,2].legend()
    # axes[1,2].grid(alpha=0.3)

    # plt.suptitle("Entropy Differences and Absolute Entropies by Outcome", fontsize=15, y=1.02)
    # plt.tight_layout()
    # plt.savefig("entropy_analysis_grid_apd.png", dpi=1000, bbox_inches="tight")