from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.isotonic import spearmanr
import torch, json, os, linecache,sys
import nethook
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
dir_path = os.path.dirname(os.path.abspath(__file__))
# print(dir_path)
sys.path.append(dir_path+"/")#add to load modules
from helper_functions import apply_whitening_batch,bootstrap_ci_spearman,bootstrap_ci_los_q1q4,_jaccard_overlap_pct,_counts
from visualizations import analyze_and_save_distances
import math
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
from torch import Tensor



#----------------------------------------------------------------------------
# Section: Qwen Direct Embeddings
#----------------------------------------------------------------------------
# region Qwen Direct Embeddings
# CONVERT THIS CODE TO HUGGING FACE VERSION
def qwen_get_embeddings(model, inputs, normalize=False):
    with torch.no_grad():
        if(len(inputs)==1):
            representation_tensor = model.encode([inputs], convert_to_tensor=True).squeeze(0)
        else:
            representation_tensor = model.encode(inputs, convert_to_tensor=True)
        if(normalize==False):
            return representation_tensor
        #else
        representation_tensor_normalized = torch.nn.functional.normalize(representation_tensor, p=2, dim=0)
        return representation_tensor_normalized

def qwen_get_embeddings_batched(model, inputs, normalize=True):

    with torch.no_grad():
        # Always pass the whole batch
        representation_tensor = model.encode(inputs, convert_to_tensor=True)

        if not normalize:
            return representation_tensor

        # Normalize each row independently (per embedding)
        representation_tensor_normalized = torch.nn.functional.normalize(
            representation_tensor, p=2, dim=1
        )# Normalize across the embedding dimension (there is no sequence dimension thus 1)
        return representation_tensor_normalized
# endregion

#----------------------------------------------------------------------------
# Section: Qwen 8b Quora PAWS
#----------------------------------------------------------------------------
#region Qwen 8b Quora PAWS
def qwen_quora_paws(data_loader,args,device):
    model = load_model_sentence_transformer(device)
    # print(model)
    # return None
    model.eval()
    incorrect_pairs=[]
    correct_pairs=[]
    neg_pairs=[]
    pos_pairs=[]
    if(args.whitening):
        whitening_stats = torch.load(args.whitening_stats_path, map_location="cpu")
    distances_file_path = os.path.join(args.save_path, "distances.jsonl")
    with open(distances_file_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="running"):
            # unpack batch
                sentences1, sentences2, labels = batch  # each is a list/tensor of size [batch_size]
                
                # get embeddings for entire batch
                embeddings1 = qwen_get_embeddings(model, sentences1)  # shape: [batch_size, hidden_dim]
                embeddings2 = qwen_get_embeddings(model, sentences2)  # shape: [batch_size, hidden_dim]
                if(args.whitening):
                    embeddings1 = apply_whitening_batch(embeddings1, whitening_stats, variant="zca", l2_after=True, device=device)
                    embeddings2 = apply_whitening_batch(embeddings2, whitening_stats, variant="zca", l2_after=True, device=device)
                # compute L2 distances for batch
                distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)  # shape: [batch_size]

                # loop over items in batch to log results
                for s1, s2, label, dist in zip(sentences1, sentences2, labels, distances):
                    dist_val = dist.item()
                    if label.item() == 0:
                        incorrect_pairs.append(dist_val)
                        neg_pairs.append((s1,s2))
                    else:
                        correct_pairs.append(dist_val)
                        pos_pairs.append((s1,s2))

                    f.write(json.dumps({
                        "sentence1": s1,
                        "sentence2": s2,
                        "label": label.item(),
                        "distance": dist_val
                    }) + "\n")
 
    _= analyze_and_save_distances(
        incorrect_pairs,
        correct_pairs,
        title_prefix="PAWS",
        out_dir=args.save_path,
        # names
        group_neg_name="Non-Paraphrase",
        group_pos_name="Paraphrase",
        # optional ROUGE text pairs
        neg_text_pairs=neg_pairs,
        pos_text_pairs=pos_pairs,
        # >>> NEW: simple styling hooks <<<
        neg_color="#1f77b4",    # blue
        pos_color="#d62728",    # red
        hist_alpha=0.35,
        kde_linewidth=2.0,
        ecdf_linewidth=2.0,
        tau_color="#2ca02c",    # green
        tau_linestyle="--",
        tau_linewidth=1.5,
        violin_facealpha=0.35,
        box_facealpha=0.35,
        )
#endregion



#----------------------------------------------------------------------------
# Section: Qwen 8b Counterfact
#----------------------------------------------------------------------------
#region Qwen 8b Counterfact
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def last_token_pool_multi(layer_hidden: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
    """
    layer_hidden: [L, B, T, H]
    attention_mask: [B, T]  (1=token, 0=pad)
    returns: [L, B, H]  (last non-pad token per sequence, per layer)
    """
    L, B, T, H = layer_hidden.shape
    # Detect left padding once for the batch
    left_padding = (attention_mask[:, -1].sum() == B)

    if left_padding:
        # Last position is always a real token
        return layer_hidden[:, :, -1, :]  # [L, B, H]
    else:
        # Index of last non-pad token for each sequence
        seq_idx = attention_mask.sum(dim=1) - 1          # [B]
        seq_idx = seq_idx.clamp(min=0).to(torch.long)    # safety

        # Build a gather index of shape [L, B, 1, H] to gather along token dim (dim=2)
        idx = seq_idx.view(1, B, 1, 1).expand(L, B, 1, H)  # [L, B, 1, H]
        pooled = layer_hidden.gather(dim=2, index=idx).squeeze(2)  # [L, B, H]
        return pooled
    
def qwen_counterfact_scpp(data_loader,args,layers,device="auto"):
    tokenizer, model = get_model_hugging_face(device)
    model.eval()
    print(model)
    
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

    l = layers+["norm"]
    
    with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
    
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
                inputs = tokenizer(
                                    anchors,
                                    padding=True,
                                    return_tensors="pt",
                                ).to(model.device)
                outputs = model(**inputs)
                anchor_sentence_embeddings=last_token_pool_multi(torch.stack([ret[layer_key].output for layer_key in ret],dim=0),inputs["attention_mask"])

                inputs = tokenizer(
                                    paraphrases,
                                    padding=True,
                                    return_tensors="pt",
                                ).to(model.device)
                outputs = model(**inputs)
                paraphrase_sentence_embeddings=last_token_pool_multi(torch.stack([ret[layer_key].output for layer_key in ret],dim=0),inputs["attention_mask"])

                inputs = tokenizer(
                                    distractors,
                                    padding=True,
                                    return_tensors="pt",
                                ).to(model.device)
                outputs = model(**inputs)
                distractor_sentence_embeddings=last_token_pool_multi(torch.stack([ret[layer_key].output for layer_key in ret],dim=0),inputs["attention_mask"])

                print(anchor_sentence_embeddings.shape,paraphrase_sentence_embeddings.shape,distractor_sentence_embeddings.shape)
    
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
    
# def qwen_counterfact_scpp(data_loader,args,device="auto"):
#     model = get_model(device)
#     model.eval()
#     # Example: access specific fields
    
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

#     file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with torch.no_grad():
#             signed_margins = []
#             jaccard_scores_list=[]
#             for batch in tqdm(data_loader, desc="Processing Rows"):
#                 # ----- batched fields -----
#                 anchors = batch["anchor"]                  # list[str]
#                 paraphrases = batch["paraphrase"]          # list[str]
#                 distractors = batch["distractor"]          # list[str]
#                 LOS_flags = batch["lexical_overlap_flag"] # list[str]
#                 scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
#                 scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
#                 scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
                

#                 # ----- encode batch (B, D) -----
#                 anchor_sentence_embeddings = model.encode(anchors, convert_to_tensor=True)       # (B, D)
#                 paraphrase_sentence_embeddings = model.encode(paraphrases, convert_to_tensor=True)# (B, D)
#                 distractor_sentence_embeddings = model.encode(distractors, convert_to_tensor=True)# (B, D)

#                 # ----- normalize row-wise -----
#                 anchor_sentence_embeddings = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=1)
#                 paraphrase_sentence_embeddings = torch.nn.functional.normalize(paraphrase_sentence_embeddings, p=2, dim=1)
#                 distractor_sentence_embeddings = torch.nn.functional.normalize(distractor_sentence_embeddings, p=2, dim=1)

#                 B = anchor_sentence_embeddings.size(0)
                

#                 # ----- per-sample pass (preserve your variable names) -----
#                 for i in range(B):
#                     anchor = anchors[i]
#                     paraphrase = paraphrases[i]
#                     distractor = distractors[i]
#                     LOS_flag = LOS_flags[i] # "low" or "high"

#                     anchor_sentence_embedding = anchor_sentence_embeddings[i]
#                     paraphrase_sentence_embedding = paraphrase_sentence_embeddings[i]
#                     distractor_sentence_embedding = distractor_sentence_embeddings[i]

#                     # --- Cosine similarities ---
#                     cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embedding, paraphrase_sentence_embedding, dim=0).item()
#                     cosine_anchor_distractor   = F.cosine_similarity(anchor_sentence_embedding, distractor_sentence_embedding, dim=0).item()
#                     cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_sentence_embedding, distractor_sentence_embedding, dim=0).item()

#                     # --- Euclidean distances ---
#                     distance_anchor_paraphrase = torch.norm(anchor_sentence_embedding - paraphrase_sentence_embedding, p=2).item()
#                     distance_anchor_distractor = torch.norm(anchor_sentence_embedding - distractor_sentence_embedding, p=2).item()
#                     distance_paraphrase_distractor = torch.norm(paraphrase_sentence_embedding - distractor_sentence_embedding, p=2).item()
                    
#                     condition_anchor = distance_anchor_distractor < distance_anchor_paraphrase 
#                     condition_paraphrase = distance_paraphrase_distractor < distance_anchor_paraphrase
#                     failure = condition_anchor or condition_paraphrase

#                     json_item={"distance_failure": failure,"lexical_overlap_flag":LOS_flag,
#                     "similarity_failure": ((cosine_anchor_distractor > cosine_anchor_paraphrase or cosine_paraphrase_distractor > cosine_anchor_paraphrase)),
#                     "distances":{"dist_cap1_cap2":distance_anchor_paraphrase,"dist_cap1_neg":distance_anchor_distractor,"dist_cap2_neg":distance_paraphrase_distractor},
#                     "similarities":{"cos_cap1_cap2":cosine_anchor_paraphrase,"cos_cap1_neg":cosine_anchor_distractor,"cos_cap2_neg":cosine_paraphrase_distractor},
#                     "anchor":anchor,"paraphrase":paraphrase,"distractor":distractor,"score_jaccard":scores_jaccard[i],"score_overlap":scores_overlap[i],"score_containment":scores_containment[i]}
#                     json.dump(json_item, jsonl_file_writer)
#                     jsonl_file_writer.write("\n")


#                     possible_margin_violation = distance_anchor_paraphrase - min(distance_paraphrase_distractor, distance_anchor_distractor)
                    
                  

#                     # correct_pairs.append(distance_anchor_paraphrase)
#                     # pos_pairs.append((anchor, paraphrase))

#                     if(LOS_flag=="low"):
#                         if(failure):
#                             jaccard_scores_list.append(scores_jaccard[i])
#                             signed_margins.append(possible_margin_violation)
#                             average_margin_violation_lor_low += possible_margin_violation
#                             failure_rate_lor_low += 1
#                         average_margin_lor_low += possible_margin_violation
#                     elif(LOS_flag=="high"):
#                         if(failure):
#                             jaccard_scores_list.append(scores_jaccard[i])
#                             signed_margins.append(possible_margin_violation)
#                             average_margin_violation_lor_high += possible_margin_violation
#                             failure_rate_lor_high += 1
#                         average_margin_lor_high += possible_margin_violation

#                     if(distance_anchor_distractor <= distance_paraphrase_distractor):
#                         incorrect_pairs.append(distance_anchor_distractor)
#                         neg_pairs.append((anchor, distractor))
#                     else:
#                         incorrect_pairs.append(distance_paraphrase_distractor)
#                         neg_pairs.append((paraphrase, distractor))

#                     correct_pairs.append(distance_anchor_paraphrase)
#                     pos_pairs.append((anchor, paraphrase))

#             rho, pval = spearmanr(jaccard_scores_list, signed_margins)
#             print("Spearman rho:", rho,pval)


#             # optional quantile split
#             signed_margins = np.array(signed_margins)   # make sure it's a NumPy array
#             jaccard_scores_list = np.array(jaccard_scores_list)
#             low_mask  = jaccard_scores_list <= np.quantile(jaccard_scores_list, 0.25)
#             high_mask = jaccard_scores_list >= np.quantile(jaccard_scores_list, 0.75)
#             LOS_sensitivity = signed_margins[high_mask].mean() - signed_margins[low_mask].mean()
#             print("LOS_sensitivity (Q1 vs Q4):", LOS_sensitivity)

#             total_samples = len(data_loader.dataset)
#             total_failures = failure_rate_lor_high + failure_rate_lor_low
#             print("total_failures", total_failures)
#             failure_rate = total_failures / total_samples if total_samples>0 else 0
#             margin_violation = average_margin_violation_lor_high + average_margin_violation_lor_low
#             avg_margin_violation = margin_violation/total_failures if total_failures>0 else 0
#             print("avg_margin_violation", avg_margin_violation)

#             rho, rho_ci = bootstrap_ci_spearman(jaccard_scores_list, signed_margins, B=10000, seed=42)
#             los, los_ci = bootstrap_ci_los_q1q4(jaccard_scores_list, signed_margins, B=10000, seed=42)

#             print(f"Spearman ρ = {rho:.3f}  (95% CI: {rho_ci[0]:.3f}, {rho_ci[1]:.3f})")
#             print(f"LOS_sensitivity (Q1–Q4) = {los:.3f}  (95% CI: {los_ci[0]:.3f}, {los_ci[1]:.3f})")
            
#             json.dump({"Failure Rate": failure_rate,
#                        "Average Margin Violation": avg_margin_violation,
#                        "spearman": [rho, pval],
#                        "LOS_sensitivity ": LOS_sensitivity}, jsonl_file_writer)
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
        
# def qwen_counterfact(data_loader,args,device,):
#     model = get_model(device)
#     model.eval()
#     # Example: access specific fields
    
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

#     file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with torch.no_grad():
#             for batch in tqdm(data_loader, desc="Processing Rows"):
    
#                 anchor, paraphrase, distractor, LOS_flag = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"][0]
#                 # print("LOS_flag",LOS_flag)
#                 # print(anchor,paraphrase,distractor )
#                 anchor_sentence_embedding = model.encode(anchor, convert_to_tensor=True).squeeze(0)
#                 paraphrase_sentence_embedding = model.encode(paraphrase, convert_to_tensor=True).squeeze(0)
#                 distractor_sentence_embedding = model.encode(distractor, convert_to_tensor=True).squeeze(0)
#                 anchor_sentence_embedding = torch.nn.functional.normalize(anchor_sentence_embedding, p=2, dim=0)
#                 paraphrase_sentence_embedding = torch.nn.functional.normalize(paraphrase_sentence_embedding, p=2, dim=0)
#                 distractor_sentence_embedding = torch.nn.functional.normalize(distractor_sentence_embedding, p=2, dim=0)
#                                 # --- Cosine similarities ---
#                 cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embedding, paraphrase_sentence_embedding, dim=0).item()
#                 cosine_anchor_distractor    = F.cosine_similarity(anchor_sentence_embedding, distractor_sentence_embedding, dim=0).item()
#                 cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_sentence_embedding, distractor_sentence_embedding, dim=0).item()

#                 # --- Euclidean distances ---
#                 distance_anchor_paraphrase = torch.norm(anchor_sentence_embedding - paraphrase_sentence_embedding, p=2).item()
#                 distance_anchor_distractor   = torch.norm(anchor_sentence_embedding - distractor_sentence_embedding, p=2).item()
#                 distance_paraphrase_distractor = torch.norm(paraphrase_sentence_embedding - distractor_sentence_embedding, p=2).item()


#                 json_item={"distance_failure": ((distance_anchor_distractor < distance_anchor_paraphrase or distance_paraphrase_distractor < distance_anchor_paraphrase)),
#                                             "similarity_failure": ((cosine_anchor_distractor > cosine_anchor_paraphrase or cosine_paraphrase_distractor > cosine_anchor_paraphrase)),
#                                             "distances":{"dist_cap1_cap2":distance_anchor_paraphrase,"dist_cap1_neg":distance_anchor_distractor,"dist_cap2_neg":distance_paraphrase_distractor},
#                                             "similarities":{"cos_cap1_cap2":cosine_anchor_paraphrase,"cos_cap1_neg":cosine_anchor_distractor,"cos_cap2_neg":cosine_paraphrase_distractor},
#                                             "anchor":anchor,"paraphrase":paraphrase,"distractor":distractor}
#                 json.dump(json_item, jsonl_file_writer)
#                 jsonl_file_writer.write("\n")

#                 condition_anchor=distance_anchor_distractor < distance_anchor_paraphrase 
#                 condition_paraphrase=distance_paraphrase_distractor < distance_anchor_paraphrase
#                 possible_margin_violation=abs(distance_anchor_paraphrase - min(distance_paraphrase_distractor,distance_anchor_distractor))
#                 failure=condition_anchor or condition_paraphrase
#                 # or (cosine_anchor_distractor > cosine_anchor_paraphrase or cosine_paraphrase_distractor > cosine_anchor_paraphrase):
#                 correct_pairs.append(distance_anchor_paraphrase)
#                 pos_pairs.append((anchor,paraphrase))
#                 if(LOS_flag=="low"):
#                     if(failure):
#                         average_margin_violation_lor_low+= possible_margin_violation#add the margin violation for failures
#                         failure_rate_lor_low+=1#increase failure count for low overlap
#                     average_margin_lor_low+= possible_margin_violation#add the margin violation general
#                 elif(LOS_flag=="high"):
#                     if(failure):
#                         average_margin_violation_lor_high+= possible_margin_violation#add the margin violation for failures
#                         failure_rate_lor_high+=1#increase failure count for high overlap
#                     average_margin_lor_high+= possible_margin_violation#add the margin violation general
                
                
#                 if(distance_anchor_distractor<=distance_paraphrase_distractor):
#                     incorrect_pairs.append(distance_anchor_distractor)
#                     neg_pairs.append((anchor,distractor))
#                 else:
#                     incorrect_pairs.append(distance_paraphrase_distractor)
#                     neg_pairs.append((paraphrase,distractor))

#                 correct_pairs.append(distance_anchor_paraphrase)
#                 pos_pairs.append((anchor,paraphrase))

                

           
                    

                   
            
#             total_samples=len(data_loader.dataset)
#             total_failures=failure_rate_lor_high+failure_rate_lor_low
#             print("total_failures",total_failures)

#             margin_violation=average_margin_violation_lor_high+average_margin_violation_lor_low
#             avg_margin_violation=margin_violation/total_failures if total_failures>0 else 0#average margin violation of the failures
#             print("avg_margin_violation",avg_margin_violation)
#             failure_rate=total_failures/total_samples#failure rate overall

#             # print("average_margin_lor_high",average_margin_lor_high)
#             LOS_sensitivity= (average_margin_lor_high/data_loader.dataset.count_high_flags) - (average_margin_lor_low/data_loader.dataset.count_low_flags)
#             print("LOS_sensitivity",LOS_sensitivity)

#             json.dump({"Failure Rate": failure_rate,"Average Margin Violation": avg_margin_violation,"LOS_sensitivity":LOS_sensitivity}, jsonl_file_writer)
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







def qwen_test_direct_counterfact_penme(file_path,model,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1
    for name, module in model.named_modules():
        print(name, ":", module.__class__.__name__)

    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    # l=["model.layers.27"]
    with open(file_save_path, 'w') as jsonl_file_writer:
        # with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
            for i in tqdm(range(5000), desc="Processing 500 steps"):
                data_dict={}
                data_entry = json.loads(linecache.getline(file_path, counter).strip())
                # print(data_entry.keys())
                torch.cuda.empty_cache()
                # print(data_entry["edited_prompt"])
                data_entry["vector_edited_prompt"]=[]
                # inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
                # outputs = model(inputs, output_hidden_states=True)
                data_dict["edit_tensor"]=model.encode([data_entry["edited_prompt"][0]], convert_to_tensor=True).squeeze(0)
                
                # print([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0].shape)
                
                # data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                torch.cuda.empty_cache()
                # data_entry["vector_edited_prompt_paraphrases_processed"]=[]
                data_dict["paraphrases_vectors"]=[]
                paraphrase_strings=[]
                # inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
                # outputs = model(inputs, output_hidden_states=True)
                data_dict["paraphrases_vectors"].append(model.encode([data_entry["edited_prompt_paraphrases_processed"]], convert_to_tensor=True).squeeze(0))
                paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed"])
                # data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                
                # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
                torch.cuda.empty_cache()
                
                # inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
                # outputs = model(inputs, output_hidden_states=True)
                # data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                data_dict["paraphrases_vectors"].append(model.encode([data_entry["edited_prompt_paraphrases_processed_testing"]], convert_to_tensor=True).squeeze(0))
                paraphrase_strings.append(data_entry["edited_prompt_paraphrases_processed_testing"])

                data_dict["locality_vectors"]=[]
                locality_strings=[]
                # data_entry["vectors_neighborhood_prompts_high_sim"]=[]
                for string in data_entry["neighborhood_prompts_high_sim"]:
                    torch.cuda.empty_cache()
                    # inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                    # outputs = model(inputs, output_hidden_states=True)
                    data_dict["locality_vectors"].append(model.encode([string], convert_to_tensor=True).squeeze(0))
                    locality_strings.append(string)
                    # data_entry["vectors_neighborhood_prompts_high_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                # data_entry["vectors_neighborhood_prompts_low_sim"]=[]
                
                for string in data_entry["neighborhood_prompts_low_sim"]:
                    torch.cuda.empty_cache()
                    # inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                    # outputs = model(inputs, output_hidden_states=True)
                    data_dict["locality_vectors"].append(model.encode([string], convert_to_tensor=True).squeeze(0))
                    locality_strings.append(string)
                    # data_entry["vectors_neighborhood_prompts_low_sim"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                for string in data_entry["openai_usable_paraphrases"]:
                    torch.cuda.empty_cache()
                    paraphrase_strings.append(string)
                    # inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
                    # outputs = model(inputs, output_hidden_states=True)
                    data_dict["paraphrases_vectors"].append(model.encode([string], convert_to_tensor=True).squeeze(0))
                # if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
                #     data_entry["openai_usable_paraphrases_embeddings"]=[]
                # data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])
        

                # Get main embedding
                edit_vec = data_dict["edit_tensor"]  # shape: [3072]

                # Stack vectors
                paraphrase_vecs = torch.stack(data_dict["paraphrases_vectors"])  # shape: [N_p, 3072]
                locality_vecs = torch.stack(data_dict["locality_vectors"])        # shape: [N_l, 3072]
                
                # edit_vec = F.normalize(edit_vec, dim=0)
                # paraphrase_vecs = F.normalize(paraphrase_vecs, dim=1)
                # locality_vecs = F.normalize(locality_vecs, dim=1)
                # print(edit_vec.shape,paraphrase_vecs.shape,locality_vecs.shape)

                # return None
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
                # min_para_dist = para_dists.min().item()
                # min_local_dist = local_dists.min().item()

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
                    break
                # if violating_pairs:       
                #     print(violating_pairs)     
                # Check comparison
                # if min_local_dist < min_para_dist or max_local_sim > max_para_sim:
                #     print("⚠️ A locality vector is closer to the edit than any paraphrase (Euclidean).",data_entry["edited_prompt"][0],paraphrase_strings[j],locality_strings[i])
                counter+=1
            # break





def qwen_test_scpp(file_path,model,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    

    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    # l=["model.layers.27"]
    with open(file_save_path, 'w') as jsonl_file_writer:
        # with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
           

            data=None
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_samples=0
            fails=0 
            for obj in tqdm(data, desc="Processing captions"):
                # print(obj)
                # print(obj["caption"],obj["caption2"],obj["negative_caption"])
                caption_tensor = model.encode([obj["caption"]], convert_to_tensor=True).squeeze(0)
                caption2_tensor = model.encode([obj["caption2"]], convert_to_tensor=True).squeeze(0)
                negative_caption_tensor = model.encode([obj["negative_caption"]], convert_to_tensor=True).squeeze(0)
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
                                            "caption":obj["caption"],"caption2":obj["caption2"],"negative_caption":obj["negative_caption"]}

                if (dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2) or (cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2):
                    fails+=1
            
                json.dump(json_item, jsonl_file_writer)
                jsonl_file_writer.write("\n")

                total_samples+=1

        print("Accuracy:", (total_samples - fails) / total_samples if total_samples > 0 else 0)
        json.dump({"Accuracy": (total_samples - fails) / total_samples if total_samples > 0 else 0  }, jsonl_file_writer)
        jsonl_file_writer.write("\n")

#___Stacked Counter fact

import json, linecache, torch
from tqdm import tqdm
import torch.nn.functional as F
# ----- Helpers -----
BATCH_ENCODE = 256        # encoding batch size for building global locality
LOCALITY_CHUNK = 8192     # chunk size for comparisons vs global locality
MAX_ROWS = None           # set to an int (e.g., 4000) to cap rows, else processes full file

device_encode = "cuda" if torch.cuda.is_available() else "cpu"
device_compute = "cpu"    # keep big comparisons on CPU to reduce GPU OOM risk

def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()

def iter_rows(file_path, max_rows=None):
    i = 1
    while True:
        if max_rows is not None and i > max_rows:
            break
        line = linecache.getline(file_path, i)
        if not line:
            break
        yield i, json.loads(line.strip())
        i += 1

def encode_strings(strings, model, batch_size=BATCH_ENCODE, device=device_encode):
    vecs = []
    with torch.no_grad():
        for b in tqdm(range(0, len(strings), batch_size), desc="Encoding locality", leave=False):
            batch = strings[b:b+batch_size]
            v = model.encode(batch, convert_to_tensor=True)
            # ensure on desired device for storage/computation staging
            v = v.to(device_compute)
            vecs.append(v)
    if len(vecs) == 0:
        return torch.empty(0, model.get_sentence_embedding_dimension(), device=device_compute)
    return torch.cat(vecs, dim=0)

def qwen_test_stacked_counterfact(file_path,model,file_save_path,device):

    # ===== PASS 1: Build global neighborhood/locality bank across the dataset =====
    global_locality_strings = []

    for _, row in tqdm(iter_rows(file_path, MAX_ROWS), desc="Pass 1: collecting locality"):
        hs = row.get("neighborhood_prompts_high_sim", []) or []
        ls = row.get("neighborhood_prompts_low_sim", []) or []
        global_locality_strings.extend(hs)
        global_locality_strings.extend(ls)

    # Optional: dedup to shrink bank
    # Using a simple set while preserving order
    seen = set()
    deduped_locality_strings = []
    for s in global_locality_strings:
        if s not in seen:
            seen.add(s)
            deduped_locality_strings.append(s)
    global_locality_strings = deduped_locality_strings

    global_locality_vecs = encode_strings(global_locality_strings, model)

    # ===== PASS 2: For each row, compare edit_tensor vs paraphrases and global locality =====
    with open(file_save_path, "w") as jsonl_file_writer, torch.no_grad():
        counter = 1
        for _, data_entry in tqdm(iter_rows(file_path, MAX_ROWS), desc="Pass 2: evaluating rows"):
            torch.cuda.empty_cache()

            # Encode edit vector
            edit_vec = model.encode([data_entry["edited_prompt"][0]], convert_to_tensor=True).squeeze(0).to(device_compute)

            # Encode paraphrases for this row
            paraphrase_strings = []
            paraphrase_vecs_list = []

            for key in ["edited_prompt_paraphrases_processed",
                        "edited_prompt_paraphrases_processed_testing"]:
                if key in data_entry:
                    paraphrase_strings.append(data_entry[key])
                    paraphrase_vecs_list.append(
                        model.encode([data_entry[key]], convert_to_tensor=True).squeeze(0).to(device_compute)
                    )

            for s in data_entry.get("openai_usable_paraphrases", []) or []:
                paraphrase_strings.append(s)
                paraphrase_vecs_list.append(
                    model.encode([s], convert_to_tensor=True).squeeze(0).to(device_compute)
                )

            if not paraphrase_vecs_list:
                continue

            paraphrase_vecs = torch.stack(paraphrase_vecs_list, dim=0)

            para_dists = torch.norm(paraphrase_vecs - edit_vec.unsqueeze(0), dim=1)
            para_sims  = F.cosine_similarity(paraphrase_vecs, edit_vec.unsqueeze(0), dim=1)
            max_para_dist = para_dists.max().item()
            min_para_sim  = para_sims.min().item()

            # === Filter global locality for this row ===
            forbidden_norm = {_norm(x) for x in paraphrase_strings + [data_entry["edited_prompt"][0]]}
            allowed_pairs = [
                (vec, s) for vec, s in zip(global_locality_vecs, global_locality_strings)
                if _norm(s) not in forbidden_norm
            ]
            if not allowed_pairs:
                continue

            locality_vecs_allowed, locality_strings_allowed = zip(*allowed_pairs)
            locality_vecs_allowed = torch.stack(locality_vecs_allowed, dim=0)

            # === Check for violations ===
            violating_entry = None
            for start in range(0, len(locality_vecs_allowed), LOCALITY_CHUNK):
                loc_chunk = locality_vecs_allowed[start:start+LOCALITY_CHUNK]
                local_dists = torch.norm(loc_chunk - edit_vec.unsqueeze(0), dim=1)
                local_sims  = F.cosine_similarity(loc_chunk, edit_vec.unsqueeze(0), dim=1)

                mask_basic = (local_dists < max_para_dist) | (local_sims > min_para_sim)
                if not torch.any(mask_basic):
                    continue

                cand_idx = torch.nonzero(mask_basic, as_tuple=False).squeeze(1)
                sel_local_dists = local_dists[cand_idx].unsqueeze(1)
                sel_local_sims  = local_sims[cand_idx].unsqueeze(1)

                cond_dist = sel_local_dists < para_dists.unsqueeze(0)
                cond_sim  = sel_local_sims  > para_sims.unsqueeze(0)
                cond = cond_dist | cond_sim

                if torch.any(cond):
                    lk_idx, p_idx = torch.nonzero(cond, as_tuple=True)
                    lk = lk_idx[0].item()
                    pj = p_idx[0].item()

                    locality_string = locality_strings_allowed[start + cand_idx[lk].item()]
                    violating_entry = {
                        "distance_failure": bool(cond_dist[lk, pj].item()),
                        "similarity_failure": bool(cond_sim[lk, pj].item()),
                        "distances_sims": {
                            "local_dist": sel_local_dists[lk, 0].item(),
                            "para_dist":  para_dists[pj].item(),
                            "local_sim":  sel_local_sims[lk, 0].item(),
                            "para_sim":   para_sims[pj].item(),
                        },
                        "edit": data_entry["edited_prompt"][0],
                        "paraphrase": paraphrase_strings[pj],
                        "locality": locality_string
                    }
                    break

            if violating_entry:
                json.dump(violating_entry, jsonl_file_writer)
                jsonl_file_writer.write("\n")
           