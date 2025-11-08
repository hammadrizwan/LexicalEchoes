
import torch, json, linecache, os
dir_path = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from helper_functions import process_and_write ,_jaccard_overlap_pct,_counts#, apply_whitening_batch,bootstrap_ci_spearman,bootstrap_ci_los_q1q4
from scipy.stats import spearmanr
import metric_functions as mf
import matplotlib.pyplot as plt
# import numpy as np
import sys, os, json
sys.path.append('/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/layer_by_layer/experiments/')
from collections import defaultdict
# import repitl.difference_of_entropies as dent
_spearman = lambda x, y: spearmanr(x, y, nan_policy="omit")
from embeddings_processing_tokenization import get_embeddings_pt,get_embedding_it
from model_loaders import get_model
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
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
#----------------------------------------------------------------------------
# Section: Gemma PT Counterfact
#----------------------------------------------------------------------------
#region Gemma PT Counterfact
def gemma_pt_counterfact_scpp(data_loader,args,access_token,layers,device="auto"):
    model,tokenizer = get_model(args.model_type,access_token,device=device)
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

    # intrasims_dict=defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing Rows"):
            anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
            # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
            scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
            scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
            scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]

            anchor_sentence_embeddings,paraphrase_sentence_embeddings,distractor_sentence_embeddings = get_embeddings_pt(model,tokenizer,args,{"anchors": anchors, "paraphrases": paraphrases, "distractors": distractors}, layers, normalize=True, device=device)

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
                    # intrasim=intrasims[layer_idx,i].item()

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
                    # intrasims_dict[layer_key].append(intrasim)
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

    process_and_write(args, layers, LO_negmax_list_by_layer, all_fail_flags_by_layer, LOc_list_by_layer, DISTc_list_by_layer, all_viols_by_layer, incorrect_pairs_by_layer, correct_pairs_by_layer, neg_pairs_by_layer, pos_pairs_by_layer)
       
#endregion         


#----------------------------------------------------------------------------
# Section: Gemma IT Counterfact
#----------------------------------------------------------------------------
#region Gemma IT Counterfact
def gemma_it_counterfact_scpp(data_loader,args,access_token,layers,device="auto"):
    model, tokenizer = get_model(args.model_type,access_token,device=device)
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
    # intrasims_dict=defaultdict(list)
    
    for batch in tqdm(data_loader, desc="Processing Rows"):

        anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
        scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
        scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
        scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
        anchor_sentence_embeddings,paraphrase_sentence_embeddings,distractor_sentence_embeddings = get_embedding_it(model,tokenizer,args, {"anchors": anchors, "paraphrases": paraphrases, "distractors": distractors}, layers, normalize=True, device=device)
        
        #_________________________________________________________________________________________

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
                # intrasim=intrasims[layer_idx,i].item()
                
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
                # intrasims_dict[layer_key].append(intrasim)
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
    process_and_write(args, layers, LO_negmax_list_by_layer, all_fail_flags_by_layer, LOc_list_by_layer, DISTc_list_by_layer, all_viols_by_layer, incorrect_pairs_by_layer, correct_pairs_by_layer, neg_pairs_by_layer, pos_pairs_by_layer)

    






# assumes you already have: load_model, build_batched_pt_inputs, nethook, mf

def gemma_counterfact_dime(
    data_loader,
    args,
    access_token,
    layers,
    device="auto",
    mode="pt",
    out_dir="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/dime_figures/",
    smooth_k=3,      # moving-average window for CCP display; set to 1 to disable
    eps=1e-8,        # small constant for CCP denominator
):
    model, tokenizer = get_model(args.model_type,access_token,device=device)
    model.eval()
    # print(model)
    # Example: access specific fields
    
    embeddings_anchors=[]
    embeddings_paraphrases=[]
    embeddings_distractors=[]

    for batch in tqdm(data_loader, desc="Processing Rows"):

        anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
        anchor_sentence_embeddings,paraphrase_sentence_embeddings,distractor_sentence_embeddings = get_embeddings_pt(model,tokenizer,args, {"anchors": anchors, "paraphrases": paraphrases, "distractors": distractors}, layers, normalize=True, device=device)
    
        embeddings_anchors.append(anchor_sentence_embeddings.detach().cpu())
        embeddings_paraphrases.append(paraphrase_sentence_embeddings.detach().cpu())
        embeddings_distractors.append(distractor_sentence_embeddings.detach().cpu())
     
    A = torch.cat(embeddings_anchors,     dim=1)  # [L, B_total, D]
    P = torch.cat(embeddings_paraphrases, dim=1)  # [L, B_total, D]
    D = torch.cat(embeddings_distractors, dim=1)  # [L, B_total, D]
    A = torch.stack([mf.normalize(A[l]) for l in range(A.shape[0])])  # [L, N, D]
    P = torch.stack([mf.normalize(P[l]) for l in range(P.shape[0])])  # [L, N, D]
    D = torch.stack([mf.normalize(D[l]) for l in range(D.shape[0])])  # [L, N, D]
    print(type(A), type(P), type(D))
    print(A.shape, P.shape, D.shape)
    print("A",A.unsqueeze(2).shape)
    print("P",P.unsqueeze(2).shape)
    print("D",D.unsqueeze(2).shape)
    # Build the two view-pairs you want to compare:
    # 1) [anchor, paraphrase]
    L, B, Demb = A.shape
    hidden_AP = torch.empty((L, B, 2, Demb), dtype=A.dtype, device=A.device)
    hidden_AP[:, :, 0, :] = A
    hidden_AP[:, :, 1, :] = P
    print(hidden_AP.shape)
    hidden_DA = torch.empty((L, B, 2, Demb), dtype=A.dtype, device=A.device)
    hidden_DA[:, :, 0, :] = A
    hidden_DA[:, :, 1, :] = D
    print(hidden_AP.shape)
    hidden_DP = torch.empty((L, B, 2, Demb), dtype=A.dtype, device=A.device)
    hidden_DP[:, :, 0, :] = P
    hidden_DP[:, :, 1, :] = D
    # Compute DiME for each pair (your compute_dime does the permute internally):
    dime_DP = mf.compute_dime(hidden_DP, alpha=1.0, normalizations=['raw'])
    print("hidden_DP.shape",dime_DP)
    dime_AP = mf.compute_dime(hidden_AP, alpha=1.0, normalizations=['raw'])
    dime_DA = mf.compute_dime(hidden_DA, alpha=1.0, normalizations=['raw'])
    # dime_DP = mf.compute_dime(hidden_DP, alpha=1.0, normalizations=['raw'])
    print("DIMES computed.")
    ap = np.array(dime_AP['raw'])
    ad = np.array(dime_DA['raw'])
    pd = np.array(dime_DP['raw'])
    layers = np.arange(len(ap))

    # Save AP DiME as JSON (for your dCor tooling)
    ap_dict = {str(int(i)): float(v) for i, v in enumerate(ap)}
    ap_json_path = os.path.join(out_dir, f"{args.model_type}_dime_AP.json")
    with open(ap_json_path, "w") as f:
        json.dump(ap_dict, f)
    print(f"[ok] wrote AP per-layer DiME (anchor–paraphrase) to {ap_json_path}")

    # --- CCP ---
    S, Lx = ap, ad
    ccp = (S - Lx) / (S + Lx + eps)

    def movavg(x, k=3):
        if k <= 1: return x
        pad = np.r_[x[0], x, x[-1]]
        return np.convolve(pad, np.ones(k)/k, mode='same')[1:-1]

    ccp_smooth = movavg(ccp, k=smooth_k)

    # summarize ccp
    peak_idx = int(np.argmax(ccp_smooth))
    first_pos_idx = int(np.argmax(ccp_smooth > 0)) if np.any(ccp_smooth > 0) else None
    print(f"[CCP] peak layer = {peak_idx}  value = {ccp_smooth[peak_idx]:.3f}")
    if first_pos_idx is not None:
        print(f"[CCP] first crossover layer = {first_pos_idx}  value = {ccp_smooth[first_pos_idx]:.3f}")
    else:
        print("[CCP] no positive crossover found")

    eps = 1e-8
    S = ap.astype(float)
    Lx = ad.astype(float)

    ccp_log = np.log((S + eps) / (Lx + eps))   # keeps rising when both grow but S grows faster


    def participation_ratio(E_np: np.ndarray) -> float:
        """
        E_np: [N, D] row=example embedding. Returns PR/D in (0,1].
        """
        if E_np.shape[0] < 2:
            return 0.0
        C = np.cov(E_np.T, bias=False)
        w = np.linalg.eigvalsh(C)
        num = (w.sum())**2
        den = (w**2).sum() + 1e-12
        pr = num / den
        return float(pr / E_np.shape[1])

    # Build a small pool per layer using A, P, D (mean-pooled, already in your tensors)
    # A, P, D are [L, N, D] torch -> stack to [N*3, D] per layer
    

    ccp_star = ccp_log  # isotropy-weighted


    fig, ax1 = plt.subplots(figsize=(12,5))
    l1, = ax1.plot(layers, ap, marker='o', label='DiME (anchor–paraphrase)')
    l2, = ax1.plot(layers, ad, marker='o', label='DiME (anchor–distractor)')
    # (optional) if you also have PD:
    # l3, = ax1.plot(layers, dp, marker='o', label='DiME (paraphrase–distractor)')
    ax1.set_xlabel("Layer"); ax1.set_ylabel("DiME (raw)")
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    l4, = ax2.plot(layers, ccp_log, linewidth=2.2, label='log-CCP (S/L)')
    # l5, = ax2.plot(layers, ccp_star, linewidth=2.2, linestyle='--', label='log-CCP × Isotropy')
    ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax2.set_ylabel("Contrast / Isotropy")

    # Mark argmax layers
    idx_log = int(np.argmax(ccp_log))
    # idx_star = int(np.argmax(ccp_star))
    ax2.scatter([idx_log], [ccp_log[idx_log]], s=50)
    # ax2.scatter([idx_star], [ccp_star[idx_star]], s=50)
    # title_marks = f"(peak log-CCP @ L{idx_log}, peak log-CCP×Iso @ L{idx_star})"
    # plt.title("DiME per Layer with log-CCP overlays  " + title_marks)

    # Merge legends
    lines = [l1, l2, l4]  # add l3 if you plotted PD
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{args.model_type}_dime_layers_AP_vs_DA_logccp.png"), dpi=900)
    # plt.show()

    # ----------------------------
    # 5) Export values for analysis
    # ----------------------------
    df_ccp = pd.DataFrame({
        "layer": layers,
        "DiME_AP": S,
        "DiME_AD": Lx,
        "log_CCP": ccp_log,
        # "Isotropy": Iso,        "log_CCP_times_Iso": ccp_star
    })
    df_ccp.to_csv(os.path.join(out_dir, f"{args.model_type}_dime_logccp_iso.csv"), index=False)
    print("[ok] saved overlays + CSV")
    # print(f"Peak log-CCP layer: {idx_log}, Peak log-CCP×Iso layer: {idx_star}")

    # --- Plot: DiME with CCP overlay (twin y-axis) ---
    plt.figure(figsize=(10,5))
    # left axis: DiME
    ax1 = plt.gca()
    line1, = ax1.plot(layers, ap, marker='o', label='DiME (anchor–paraphrase)')
    line2, = ax1.plot(layers, ad, marker='o', label='DiME (anchor–distractor)')
    line3, = ax1.plot(layers, pd, marker='o', label='DiME (paraphrase–distractor)')
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("DiME (raw)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # right axis: CCP
    ax2 = ax1.twinx()
    line4, = ax2.plot(layers, ccp_smooth, linewidth=2.2, alpha=0.85, label='CCP (smoothed)')
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("CCP")

    # mark first crossover + peak on CCP
    marks = []
    if first_pos_idx is not None:
        ax2.scatter([first_pos_idx], [ccp_smooth[first_pos_idx]], s=60, zorder=5)
        marks.append(f"1st cross @ L{first_pos_idx}")
    ax2.scatter([peak_idx], [ccp_smooth[peak_idx]], s=60, zorder=5)
    marks.append(f"peak @ L{peak_idx}")

    title_suffix = "  (" + ", ".join(marks) + ")"
    plt.title("DiME per Layer with CCP overlay" + title_suffix)

    # combine legends
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    dime_ccp_png = os.path.join(out_dir, f"{args.model_type}_dime_layers_AP_vs_DA.png")
    plt.savefig(dime_ccp_png, dpi=1000)
    print(f"[ok] saved figure with CCP overlay to {dime_ccp_png}")

    # --- Optional: standalone CCP plot ---
    plt.figure(figsize=(10,4))
    plt.plot(layers, ccp_smooth, label="CCP (smoothed)")
    plt.plot(layers, ccp, alpha=0.35, linewidth=1, label="CCP (raw)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    if first_pos_idx is not None:
        plt.scatter([first_pos_idx], [ccp_smooth[first_pos_idx]], s=60)
        plt.text(first_pos_idx, ccp_smooth[first_pos_idx], " 1st cross", va="bottom")
    plt.scatter([peak_idx], [ccp_smooth[peak_idx]], s=60)
    plt.text(peak_idx, ccp_smooth[peak_idx], " peak", va="bottom")
    plt.title("CCP across layers")
    plt.xlabel("Layer")
    plt.ylabel("CCP = (DiME(AP) - DiME(AD)) / (DiME(AP)+DiME(AD))")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    ccp_png = os.path.join(out_dir, f"{args.model_type}_ccp_layers.png")
    plt.savefig(ccp_png, dpi=600)
    print(f"[ok] saved CCP-only figure to {ccp_png}")

    # # Return arrays for any downstream analysis
    # return {
    #     "layers": layers_idx,
    #     "dime_AP": ap,
    #     "dime_AD": ad,
    #     "dime_PD": pd,
    #     "ccp": ccp,
    #     "ccp_smooth": ccp_smooth,
    #     "peak_layer": peak_idx,
    #     "first_crossover_layer": first_pos_idx,
    #     "csv_path": ccp_csv,
    #     "fig_path_dime_ccp": dime_ccp_png,
    #     "fig_path_ccp": ccp_png,
    # }

# def gemma_counterfact_dime(data_loader,args,access_token,layers,device="auto",mode="pt"):
#     tokenizer,model = load_model(args.model_type,access_token,device=device)
#     model.eval()
#     # Example: access specific fields
    
#     embeddings_anchors=[]
#     embeddings_paraphrases=[]
#     embeddings_distractors=[]
#     l = layers
    
#     with nethook.TraceDict(model, l) as ret:
#         with torch.no_grad():
#             for batch in tqdm(data_loader, desc="Processing Rows"):
   
#                 anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
            
              
#                 inputs = build_batched_pt_inputs(tokenizer, anchors, device=device)
#                 _ = model(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     output_hidden_states=True
#                 ) 
#                 anchor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
#                 mask = inputs["attention_mask"].unsqueeze(-1)
                
#                 padded_zero_embeddings=anchor_sentence_embeddings * mask
#                 sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
#                 lengths = mask.sum(dim=1)                                  # (B, L, 1)
#                 anchor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
                 
#                 inputs = build_batched_pt_inputs(tokenizer, paraphrases, device=device)
#                 _ = model(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     output_hidden_states=True
#                 ) 
#                 paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
#                 mask = inputs["attention_mask"].unsqueeze(-1)
#                 padded_zero_embeddings=paraphrase_sentence_embeddings * mask
#                 sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
#                 lengths = mask.sum(dim=1)                                     # (B, ,L, 1)
#                 paraphrase_sentence_embeddings = sum_hidden / lengths.clamp(min=1)
#                 inputs = build_batched_pt_inputs(tokenizer, distractors, device=device)
#                 _ = model(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     output_hidden_states=True
#                 ) 
#                 distractor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
#                 mask = inputs["attention_mask"].unsqueeze(-1)
              
#                 padded_zero_embeddings=distractor_sentence_embeddings * mask
#                 sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
#                 lengths = mask.sum(dim=1)                                     # (B, ,L, 1)
#                 distractor_sentence_embeddings = sum_hidden / lengths.clamp(min=1)

            
#                 embeddings_anchors.append(anchor_sentence_embeddings.detach().cpu())
#                 embeddings_paraphrases.append(paraphrase_sentence_embeddings.detach().cpu())
#                 embeddings_distractors.append(distractor_sentence_embeddings.detach().cpu())

                
                
#         A = torch.cat(embeddings_anchors,     dim=1)  # [L, B_total, D]
#         P = torch.cat(embeddings_paraphrases, dim=1)  # [L, B_total, D]
#         D = torch.cat(embeddings_distractors, dim=1)  # [L, B_total, D]
#         A = torch.stack([mf.normalize(A[l]) for l in range(A.shape[0])])  # [L, N, D]
#         P = torch.stack([mf.normalize(P[l]) for l in range(P.shape[0])])  # [L, N, D]
#         D = torch.stack([mf.normalize(D[l]) for l in range(D.shape[0])])  # [L, N, D]
#         print(type(A), type(P), type(D))
#         print(A.shape, P.shape, D.shape)
#         print("A",A.unsqueeze(2).shape)
#         print("P",P.unsqueeze(2).shape)
#         print("D",D.unsqueeze(2).shape)
#         # Build the two view-pairs you want to compare:
#         # 1) [anchor, paraphrase]
#         L, B, Demb = A.shape
#         hidden_AP = torch.empty((L, B, 2, Demb), dtype=A.dtype, device=A.device)
#         hidden_AP[:, :, 0, :] = A
#         hidden_AP[:, :, 1, :] = P
#         print(hidden_AP.shape)
#         hidden_DA = torch.empty((L, B, 2, Demb), dtype=A.dtype, device=A.device)
#         hidden_DA[:, :, 0, :] = A
#         hidden_DA[:, :, 1, :] = D
#         print(hidden_AP.shape)
#         hidden_DP = torch.empty((L, B, 2, Demb), dtype=A.dtype, device=A.device)
#         hidden_DP[:, :, 0, :] = P
#         hidden_DP[:, :, 1, :] = D
#         # Compute DiME for each pair (your compute_dime does the permute internally):
#         dime_AP = mf.compute_dime(hidden_AP, alpha=1.0, normalizations=['raw'])
#         dime_DA = mf.compute_dime(hidden_DA, alpha=1.0, normalizations=['raw'])
#         dime_DP = mf.compute_dime(hidden_DP, alpha=1.0, normalizations=['raw'])
#         print("DIMES computed.")
#         ap = np.array(dime_AP['raw'])
#         da = np.array(dime_DA['raw'])
#         dp = np.array(dime_DP['raw'])
#         layers = np.arange(len(ap))

#         #############################################
#         # SAVE 'ap' IN dCor-COMPATIBLE FORMAT
#         #############################################

#         # Build { "0": <ap_at_layer_0>, "1": <ap_at_layer_1>, ... }
#         ap_dict = {
#             str(int(layer_idx)): float(val)
#             for layer_idx, val in enumerate(ap)
#         }

#         out_dir = "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/dime_figures/"
#         os.makedirs(out_dir, exist_ok=True)

#         ap_json_path = os.path.join(
#             out_dir,
#             f"{args.model_type}_dime_AP.json"  # <-- this is what you'll feed into dCor code
#         )

#         with open(ap_json_path, "w") as f:
#             json.dump(ap_dict, f)  # no indent needed for loading

#         print(f"[ok] wrote AP per-layer DiME (anchor–paraphrase) to {ap_json_path}")
# #############################################


#         plt.figure(figsize=(10,5))
#         plt.plot(layers, ap, marker='o', label='DiME (anchor–paraphrase)')
#         plt.plot(layers, da, marker='o', label='DiME (anchor–distractor)')
#         plt.plot(layers, dp, marker='o', label='DiME (paraphrase–distractor)')
#         plt.xlabel("Layer")
#         plt.ylabel("DiME (raw)")
#         plt.title("DiME per Layer")
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.savefig(f"/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/dime_figures/{args.model_type}_dime_layers_AP_vs_DA.png", dpi=1000)
#         # plt.show()

        
#         InfoNCE_AP = mf.compute_infonce(hidden_AP)
#         InfoNCE_DA = mf.compute_infonce(hidden_DA)
#         InfoNCE_DP = mf.compute_infonce(hidden_DP)
#         print("DIMES computed.")
#         ap = np.array(InfoNCE_AP['raw'])
#         da = np.array(InfoNCE_DA['raw'])
#         dp = np.array(InfoNCE_DP['raw'])
#         layers = np.arange(len(ap))

#         plt.figure(figsize=(10,5))
#         plt.plot(layers, ap, marker='o', label='InfoNCE (anchor–paraphrase)')
#         plt.plot(layers, da, marker='o', label='InfoNCE (anchor–distractor)')
#         plt.plot(layers, dp, marker='o', label='InfoNCE (paraphrase–distractor)')
#         plt.xlabel("Layer")
#         plt.ylabel("InfoNCE (raw)")
#         plt.title("InfoNCE per Layer")
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.savefig(f"/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/dime_figures/{args.model_type}_InfoNCE_layers_AP_vs_DA.png", dpi=1000)


# def gemma_test_direct_counterfact_easyedit(file_path,model,file_save_path,tokenizer,device):
#     # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     counter=1
#     for name, module in model.named_modules():
#         print(name, ":", module.__class__.__name__)

 

#     # Read the JSON file
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     # If it's a list of dictionaries, convert to DataFrame
#     df = pd.DataFrame(data)


#     # Example: access specific fields
#     total_samples=0
#     fails=0
#     l=["model.layers.27"]
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with nethook.TraceDict(model, l) as ret:
#             with torch.no_grad():
#                 for row in tqdm(data, desc="Processing rows"):
#                     inputs=tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     caption_tensor=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]

#                     inputs=tokenizer(row["rephrase_prompt"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     caption2_tensor=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]


#                     inputs=tokenizer(row["locality_prompt"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     negative_caption_tensor=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]

#                     # caption_tensor = model.encode([row["prompt"]], convert_to_tensor=True).squeeze(0)
#                     # caption2_tensor = model.encode([row["rephrase_prompt"]], convert_to_tensor=True).squeeze(0)
#                     # negative_caption_tensor = model.encode([row["locality_prompt"]], convert_to_tensor=True).squeeze(0)
#                     caption_tensor = torch.nn.functional.normalize(caption_tensor, p=2, dim=0)
#                     caption2_tensor = torch.nn.functional.normalize(caption2_tensor, p=2, dim=0)
#                     negative_caption_tensor = torch.nn.functional.normalize(negative_caption_tensor, p=2, dim=0)
#                                     # --- Cosine similarities ---
#                     cos_cap1_cap2   = F.cosine_similarity(caption_tensor, caption2_tensor, dim=0).item()
#                     cos_cap1_neg    = F.cosine_similarity(caption_tensor, negative_caption_tensor, dim=0).item()
#                     cos_cap2_neg    = F.cosine_similarity(caption2_tensor, negative_caption_tensor, dim=0).item()

#                     # --- Euclidean distances ---
#                     dist_cap1_cap2  = torch.norm(caption_tensor - caption2_tensor, p=2).item()
#                     dist_cap1_neg   = torch.norm(caption_tensor - negative_caption_tensor, p=2).item()
#                     dist_cap2_neg   = torch.norm(caption2_tensor - negative_caption_tensor, p=2).item()


#                     json_item={"distance_failure": ((dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2)),
#                                                 "similarity_failure": ((cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2)),
#                                                 "distances":{"dist_cap1_cap2":dist_cap1_cap2,"dist_cap1_neg":dist_cap1_neg,"dist_cap2_neg":dist_cap2_neg},
#                                                 "similarities":{"cos_cap1_cap2":cos_cap1_cap2,"cos_cap1_neg":cos_cap1_neg,"cos_cap2_neg":cos_cap2_neg},
#                                                 "caption":row["prompt"],"caption2":row["rephrase_prompt"],"negative_caption":row["locality_prompt"]}

#                     if (dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2) or (cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2):
#                         fails+=1
                
#                         json.dump(json_item, jsonl_file_writer)
#                         jsonl_file_writer.write("\n")

#                     total_samples+=1

#             print("Accuracy:", (total_samples - fails) / total_samples if total_samples > 0 else 0)
#             json.dump({"Accuracy": (total_samples - fails) / total_samples if total_samples > 0 else 0  }, jsonl_file_writer)
#             jsonl_file_writer.write("\n")
                

# def gemma_embeddings_analysis_counterfact_lasttoken(file_path,model,tokenizer,file_save_path,device):
#     # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     counter=1

#     # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
#     l=["model.layers.27"]
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with nethook.TraceDict(model, l) as ret:
#             with torch.no_grad():
#                 # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
#                 for i in tqdm(range(500), desc="Processing 500 steps"):
#                     data_dict={}
#                     data_entry = json.loads(linecache.getline(file_path, counter).strip())
#                     # print(data_entry.keys())
#                     torch.cuda.empty_cache()
#                     # print(data_entry["edited_prompt"])
#                     data_entry["vector_edited_prompt"]=[]
#                     inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     data_dict["edit_tensor"]=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]
                    
#                     # print([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0].shape)
                    
#                     # data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
#                     torch.cuda.empty_cache()
#                     # data_entry["vector_edited_prompt_paraphrases_processed"]=[]
#                     data_dict["paraphrases_vectors"]=[]
#                     paraphrase_strings=[]
#                     inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
#                     paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed"])
#                     # data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    
#                     # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
#                     torch.cuda.empty_cache()
                    
#                     inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
#                     data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
#                     paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed_testing"])

#                     data_dict["locality_vectors"]=[]
#                     locality_strings=[]
#                     # data_entry["vectors_neighborhood_prompts_high_sim"]=[]
#                     for string in data_entry["neighborhood_prompts_high_sim"]:
#                         torch.cuda.empty_cache()
#                         inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
#                         outputs = model(inputs, output_hidden_states=True)
#                         data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
#                         locality_strings.append(string)
#                         # data_entry["vectors_neighborhood_prompts_high_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

#                     # data_entry["vectors_neighborhood_prompts_low_sim"]=[]
                    
#                     for string in data_entry["neighborhood_prompts_low_sim"]:
#                         torch.cuda.empty_cache()
#                         inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
#                         outputs = model(inputs, output_hidden_states=True)
#                         data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
#                         locality_strings.append(string)
#                         # data_entry["vectors_neighborhood_prompts_low_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

#                     for string in data_entry["openai_usable_paraphrases"]:
#                         torch.cuda.empty_cache()
#                         paraphrase_strings.append(string)
#                         inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
#                         outputs = model(inputs, output_hidden_states=True)
#                         data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0])
#                     # if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
#                     #     data_entry["openai_usable_paraphrases_embeddings"]=[]
#                     # data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])
                


#                     # Get main embedding
#                     edit_vec = data_dict["edit_tensor"]  # shape: [3072]

#                     # Stack vectors
#                     paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # shape: [N_p, 3072]
#                     locality_vecs = torch.stack(data_dict["locality_vectors"])        # shape: [N_l, 3072]
                    
#                     edit_vec = F.normalize(edit_vec, dim=0)
#                     paraphrase_vecs = F.normalize(paraphrase_vecs, dim=1)
#                     locality_vecs = F.normalize(locality_vecs, dim=1)
#                     # 1. Distances from edit → each paraphrase
#                     para_sims = F.cosine_similarity(
#                         paraphrase_vecs, edit_vec.unsqueeze(0), dim=1
#                     )  # shape: [N_p], similarity scores

#                     # 2. Distances from edit → each locality vector
#                     local_sims = F.cosine_similarity(
#                         locality_vecs, edit_vec.unsqueeze(0), dim=1
#                     )  # shape: [N_l], similarity scores

#                     # # 3. Compare
#                     # max_para_sim = para_distances.max().item()  # highest similarity
#                     # max_local_sim = local_distances.max().item()

#                     # print(f"Closest paraphrase sim: {max_para_sim:.4f}")
#                     # print(f"Closest locality sim:  {max_local_sim:.4f}")
#                     # counter+=1
#                         # json.dump(data_entry, jsonl_file_writer)
#                         # jsonl_file_writer.write('\n')

#                     # Ensure everything is on the same device
#                     edit_vec = data_dict["edit_tensor"]  # shape: [3072]
#                     paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # [N_p, 3072]
#                     locality_vecs = torch.stack(data_dict["locality_vectors"])        # [N_l, 3072]

#                     edit_vec = edit_vec.to(paraphrase_vecs.device)

#                     # Compute Euclidean distances
#                     para_dists = torch.norm(paraphrase_vecs - edit_vec.unsqueeze(0), dim=1)  # [N_p]
#                     local_dists = torch.norm(locality_vecs - edit_vec.unsqueeze(0), dim=1)   # [N_l]

#                     # Min distances
#                     min_para_dist = para_dists.min().item()
#                     min_local_dist = local_dists.min().item()

#                     # Print for debug
#                     # print(f"Closest paraphrase distance: {min_para_dist:.4f}")
#                     # print(f"Closest locality distance:   {min_local_dist:.4f}")
#                     violating_pairs=[]
#                     for i, (local_dist,local_sim) in enumerate(zip(local_dists,local_sims)):
#                         for j, (para_dist, para_sim ) in enumerate(zip(para_dists,para_sims)):
#                             if local_dist < para_dist or local_sim > para_sim:
#                                 # print("heelo")
#                                 violating_pairs.append({"distance_failure": (local_dist < para_dist).item(),"similarity_failure": (local_sim > para_sim).item(),"distances_sims":{"local_dist":local_dist.item(),"para_dist":para_dist.item(),"local_sim":local_sim.item(),"para_sim":para_sim.item()},"edit":data_entry["edited_prompt"][0],"paraphrase":paraphrase_strings[j],"locality":locality_strings[i]})
                    
                    
#                     for entry in violating_pairs:
#                         # print(entry)
#                         json.dump(entry, jsonl_file_writer)
#                         jsonl_file_writer.write("\n")
#                     # if violating_pairs:       
#                     #     print(violating_pairs)     
#                     # Check comparison
#                     # if min_local_dist < min_para_dist or max_local_sim > max_para_sim:
#                     #     print("⚠️ A locality vector is closer to the edit than any paraphrase (Euclidean).",data_entry["edited_prompt"][0],paraphrase_strings[j],locality_strings[i])
#                     counter+=1
#                 # break


# def gemma_embeddings_analysis_counterfact_average(file_path,model,tokenizer,file_save_path,device):
#     # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     counter=1

#     # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
#     l=["model.layers.27"]
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with nethook.TraceDict(model, l) as ret:
#             with torch.no_grad():
#                 # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
#                 for i in tqdm(range(500), desc="Processing 500 steps"):
#                     data_dict={}
#                     data_entry = json.loads(linecache.getline(file_path, counter).strip())
#                     # print(data_entry.keys())
#                     torch.cuda.empty_cache()
#                     # print(data_entry["edited_prompt"])
#                     data_entry["vector_edited_prompt"]=[]
#                     inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     data_dict["edit_tensor"]=[ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0]
                   
#                     # print([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0].shape)
                    
#                     # data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
#                     torch.cuda.empty_cache()
#                     # data_entry["vector_edited_prompt_paraphrases_processed"]=[]
#                     data_dict["paraphrases_vectors"]=[]
#                     paraphrase_strings=[]
#                     inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
#                     paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed"])
#                     # data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    
#                     # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
#                     torch.cuda.empty_cache()
                    
#                     inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
#                     outputs = model(inputs, output_hidden_states=True)
#                     # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
#                     data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
#                     paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed_testing"])

#                     data_dict["locality_vectors"]=[]
#                     locality_strings=[]
#                     # data_entry["vectors_neighborhood_prompts_high_sim"]=[]
#                     for string in data_entry["neighborhood_prompts_high_sim"]:
#                         torch.cuda.empty_cache()
#                         inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
#                         outputs = model(inputs, output_hidden_states=True)
#                         data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
#                         locality_strings.append(string)
#                         # data_entry["vectors_neighborhood_prompts_high_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

#                     # data_entry["vectors_neighborhood_prompts_low_sim"]=[]
                    
#                     for string in data_entry["neighborhood_prompts_low_sim"]:
#                         torch.cuda.empty_cache()
#                         inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
#                         outputs = model(inputs, output_hidden_states=True)
#                         data_dict["locality_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
#                         locality_strings.append(string)
#                         # data_entry["vectors_neighborhood_prompts_low_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

#                     for string in data_entry["openai_usable_paraphrases"]:
#                         torch.cuda.empty_cache()
#                         paraphrase_strings.append(string)
#                         inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
#                         outputs = model(inputs, output_hidden_states=True)
#                         data_dict["paraphrases_vectors"].append([ret[layer_fc1_vals].output[0].mean(dim=1).view(-1) for layer_fc1_vals in ret][0])
#                     # if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
#                     #     data_entry["openai_usable_paraphrases_embeddings"]=[]
#                     # data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])
            

#                     # Get main embedding
#                     edit_vec = data_dict["edit_tensor"]  # shape: [3072]

#                     # Stack vectors
#                     paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # shape: [N_p, 3072]
#                     locality_vecs = torch.stack(data_dict["locality_vectors"])        # shape: [N_l, 3072]
                    
#                     edit_vec = F.normalize(edit_vec, dim=0)
#                     paraphrase_vecs = F.normalize(paraphrase_vecs, dim=1)
#                     locality_vecs = F.normalize(locality_vecs, dim=1)
#                     # 1. Distances from edit → each paraphrase
#                     para_sims = F.cosine_similarity(
#                         paraphrase_vecs, edit_vec.unsqueeze(0), dim=1
#                     )  # shape: [N_p], similarity scores

#                     # 2. Distances from edit → each locality vector
#                     local_sims = F.cosine_similarity(
#                         locality_vecs, edit_vec.unsqueeze(0), dim=1
#                     )  # shape: [N_l], similarity scores

#                     # # 3. Compare
#                     # max_para_sim = para_distances.max().item()  # highest similarity
#                     # max_local_sim = local_distances.max().item()

#                     # print(f"Closest paraphrase sim: {max_para_sim:.4f}")
#                     # print(f"Closest locality sim:  {max_local_sim:.4f}")
#                     # counter+=1
#                         # json.dump(data_entry, jsonl_file_writer)
#                         # jsonl_file_writer.write('\n')

#                     # Ensure everything is on the same device
#                     edit_vec = data_dict["edit_tensor"]  # shape: [3072]
#                     paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # [N_p, 3072]
#                     locality_vecs = torch.stack(data_dict["locality_vectors"])        # [N_l, 3072]

#                     edit_vec = edit_vec.to(paraphrase_vecs.device)

#                     # Compute Euclidean distances
#                     para_dists = torch.norm(paraphrase_vecs - edit_vec.unsqueeze(0), dim=1)  # [N_p]
#                     local_dists = torch.norm(locality_vecs - edit_vec.unsqueeze(0), dim=1)   # [N_l]

#                     # Min distances
#                     min_para_dist = para_dists.min().item()
#                     min_local_dist = local_dists.min().item()

#                     # Print for debug
#                     # print(f"Closest paraphrase distance: {min_para_dist:.4f}")
#                     # print(f"Closest locality distance:   {min_local_dist:.4f}")
#                     violating_pairs=[]
#                     for i, (local_dist,local_sim) in enumerate(zip(local_dists,local_sims)):
#                         for j, (para_dist, para_sim ) in enumerate(zip(para_dists,para_sims)):
#                             if local_dist < para_dist or local_sim > para_sim:
#                                 # print("heelo")
#                                 violating_pairs.append({"distance_failure": (local_dist < para_dist).item(),"similarity_failure": (local_sim > para_sim).item(),"distances_sims":{"local_dist":local_dist.item(),"para_dist":para_dist.item(),"local_sim":local_sim.item(),"para_sim":para_sim.item()},"edit":data_entry["edited_prompt"][0],"paraphrase":paraphrase_strings[j],"locality":locality_strings[i]})
                    
                    
#                     for entry in violating_pairs:
#                         json.dump(entry, jsonl_file_writer)
#                         jsonl_file_writer.write("\n")
#                     # if violating_pairs:       
#                     #     print(violating_pairs)     
#                     # Check comparison
#                     # if min_local_dist < min_para_dist or max_local_sim > max_para_sim:
#                     #     print("⚠️ A locality vector is closer to the edit than any paraphrase (Euclidean).",data_entry["edited_prompt"][0],paraphrase_strings[j],locality_strings[i])
#                     counter+=1
#                 # break
# def compute_intra_sentence_similarity(embeddings, mask, mean_norm, eps=1e-12,average_over_batch=False):
#     """
#     Compute IntraSimᶩ(s) = (1/n) Σ_i cos(~sᶩ, fᶩ(s,i))
#     for each layer and batch.

#     Args:
#         embeddings: Tensor [L, B, T, H]
#             Layerwise token embeddings.
#         mask: Tensor [B, T, 1]
#             1 for valid tokens, 0 for padding.
#         mean_norm: Tensor [L, B, H]
#             Normalized mean embedding per layer and batch.
#         eps: float
#             Small epsilon for numerical stability.

#     Returns:
#         intra_sim: Tensor [L, B]
#             Intra-sentence similarity per layer and batch.
#     """
#      # Normalize token embeddings (safe even with zeroed padding)
#     emb_norm = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(eps)  # [L, B, T, H]

#     # Expand mean embedding to match token dimension
#     mean_norm_exp = mean_norm.unsqueeze(2)  # [L, B, 1, H]

#     # Cosine similarities for each token
#     cos_sim = (emb_norm * mean_norm_exp).sum(dim=-1)  # [L, B, T]

#     # Average across valid tokens
#     valid_counts = mask.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
#     print("valid_counts",valid_counts.transpose(0, 1))
#     intra_sim = cos_sim.sum(dim=2) / valid_counts.transpose(0, 1)  # [L, B]
#     print("intra_sim",intra_sim)
#     # Optionally average across batch
#     if average_over_batch:
#         intra_sim = intra_sim.mean(dim=1)  # [L]

#     return intra_sim