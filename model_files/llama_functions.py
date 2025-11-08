
import torch
import nethook
import json,os
from tqdm import tqdm
import linecache
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from visualizations import analyze_and_save_distances
import numpy as np
from helper_functions import _counts, _jaccard_overlap_pct,matrix_entropy_batch_from_tokens, process_and_write
import random
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from embeddings_processing_tokenization import get_embedding_it, get_embeddings_pt
from collections import defaultdict
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
from model_loaders import get_model

#----------------------------------------------------------------------------
# Section: Llama model Instruct Analysis Counterfact
#----------------------------------------------------------------------------
#region Llama model Instruct Analysis Counterfact




# def to_1d_numpy(x):
#     try:
#         import torch
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
#     except ImportError:
#         pass
#     x = np.asarray(x).reshape(-1)
#     return x
# def to_np(x):
#     try:
#         import torch
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
#     except ImportError:
#         pass
#     return np.asarray(x)

# def summarize(arr, mask, name):
#     m = np.mean(arr[mask]) if mask.any() else np.nan
#     s = np.std(arr[mask]) if mask.any() else np.nan
#     q = np.percentile(arr[mask], [25,50,75]) if mask.any() else [np.nan]*3
#     print(f"{name}: mean={m:.4f} std={s:.4f} q25/50/75={q}")

# def select_topn_dims_by_variance(emb, mask,percentage_selection):
#     """
#     emb:  (B, T, H)  - full embeddings
#     mask: (B, T, 1)  - 1 for valid tokens, 0 for padding (broadcastable over H)
#     n:    int        - number of feature dimensions to keep (per batch element)

#     Returns:
#       emb_reduced: (B, T, n)  - embeddings restricted to top-n dims per sample
#       topn_idx:    (B, n)     - indices of selected dims in original H
#       var:         (B, H)     - variance used for selection
#     """
#     B, T, H = emb.shape
#     # Zero-out pads
#     # emb_z = emb * mask  # (B, T, H)
#     # print("mask",mask.shape,"embz",emb.shape)
#     # Per-sequence mean across time (ignore pads)
#     lengths = mask.sum(dim=1).clamp(min=1)              # (B, 1, 1)
#     mean = emb.sum(dim=1) / lengths     # (B, 1, H)
#     # print("--------------------\nmean",mean.shape,lengths.shape)
#     # Per-sequence variance across time (ignore pads)
#     sq_diff = ((emb - mean.unsqueeze(1)) ** 2) * mask           # (B, T, H)
#     # print("sq_diff",sq_diff.shape)
#     var = sq_diff.sum(dim=1) / lengths     # (B, H)
#     k=int(H*percentage_selection)
#     # Top-n dims per batch element
#     topn_idx = var.topk(k, dim=1, largest=True).indices               # (B, n)

#     # Gather those dims for every timestep in that sample
#     idx_expanded = topn_idx.unsqueeze(1).expand(-1, T, -1)  # (B, T, n)

#     emb_reduced = torch.gather(emb, dim=2, index=idx_expanded)  # (B, T, n)
#     # print("emb_reduced",emb_reduced.shape)
#     return emb_reduced, topn_idx, var

def llama_it_counterfact_scpp(data_loader,args,acess_token,layers,device):
    """
    Run PAWS-like pair distance extraction on Llama 3.2 Instruct checkpoints.
    Expects:
      - args.model_type in {"Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"} (without 'meta-llama/' prefix)
      - get_model(), acess_token_gemma, nethook, etc. defined in your env (mirrors your Gemma code)
      - args.save_path exists
    """
    import os, json, torch
    from tqdm import tqdm



    print("Loading model...")
    model,tokenizer  = get_model(args.model_type, acess_token, device)
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

    for batch in tqdm(data_loader, desc="Processing Rows"):

        anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
        # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
        scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
        scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
        scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
        anchor_sentence_embeddings,paraphrase_sentence_embeddings, distractor_sentence_embeddings = get_embedding_it(model, tokenizer, args, {"anchor": anchors, "paraphrase": paraphrases, "distractor": distractors}, layers, normalize=True ,device=device)
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

    process_and_write(args, layers, LO_negmax_list_by_layer, all_fail_flags_by_layer, LOc_list_by_layer, DISTc_list_by_layer, all_viols_by_layer, incorrect_pairs_by_layer, correct_pairs_by_layer, neg_pairs_by_layer, pos_pairs_by_layer)
#endregion
# region Llama model Pretrained Analysis

def llama_pt_counterfact_scpp(data_loader,args,access_token,layers,device="auto"):
    model,tokenizer = get_model(args.model_type,access_token,device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    # Example: access specific fields
    print(model)
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
    print(layers)

    for batch in tqdm(data_loader, desc="Processing Rows"):

        anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
        # anchors, paraphrases, distractors = batch["anchor"], batch["paraphrase"], batch["distractor"]
        scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
        scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
        scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
        # Sentence 1
        anchor_sentence_embeddings,paraphrase_sentence_embeddings, distractor_sentence_embeddings = get_embeddings_pt(model, tokenizer, args, {"anchors": anchors, "paraphrases": paraphrases, "distractors": distractors}, layers, normalize=True ,device=device)
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

    process_and_write(args, layers, LO_negmax_list_by_layer, all_fail_flags_by_layer, LOc_list_by_layer, DISTc_list_by_layer, all_viols_by_layer, incorrect_pairs_by_layer, correct_pairs_by_layer, neg_pairs_by_layer, pos_pairs_by_layer)

#endregion
#     
# def llama_pt_counterfact_scpp(data_loader,args,acess_token,layer_mapping_dict,device):
#     """
#     Run PAWS-like pair distance extraction on Llama 3.2 Instruct checkpoints.
#     Expects:
#       - args.model_type in {"Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"} (without 'meta-llama/' prefix)
#       - get_model(), acess_token_gemma, nethook, etc. defined in your env (mirrors your Gemma code)
#       - args.save_path exists
#     """



#     print("Loading model...")
#     tokenizer, model = get_model(args.model_type, acess_token, device)
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

# def llama_embeddings_analysis_counterfact_lasttoken(file_path,model,tokenizer,file_save_path,device):
#     # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     counter=1

#     # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
#     l=["model.layers.31"]
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


# def llama_embeddings_analysis_counterfact_average(file_path,model,tokenizer,file_save_path,device):
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


#     # apd = to_1d_numpy(comparison_entropies_apd)
#     # ap  = to_1d_numpy(comparison_entropies_ap)
#     # flags = to_1d_numpy(all_fail_flags).astype(int)
#     # fail_mask = flags == 1
#     # succ_mask = flags == 0
#     # H_A = to_np(entropies_anchor)      # list/array of anchor entropies
#     # H_P = to_np(entropies_paraphrase)  # list/array of paraphrase entropies
#     # H_D = to_np(entropies_distractor) 
#     # correct_pairs=to_np(correct_pairs)
#     # summarize(apd, succ_mask, "APD Success"); summarize(apd, fail_mask, "APD Failure")
#     # summarize(ap,  succ_mask, "AP  Success"); summarize(ap,  fail_mask, "AP  Failure")
#     # fig, axes = plt.subplots(2, 3, figsize=(14, 8))

#     # # --- top row: scatter plots ---
#     # axes[0,0].scatter(ap[succ_mask], apd[succ_mask], s=25, alpha=0.6, color="tab:blue")
#     # axes[0,0].set_title("Success Cases")
#     # axes[0,0].set_xlabel("ap")
#     # axes[0,0].set_ylabel("apd")
#     # axes[0,0].grid(alpha=0.3)

#     # axes[0,1].scatter(ap[fail_mask], apd[fail_mask], s=25, alpha=0.6, color="tab:orange")
#     # axes[0,1].set_title("Failure Cases")
#     # axes[0,1].set_xlabel("ap")
#     # axes[0,1].set_ylabel("apd")
#     # axes[0,1].grid(alpha=0.3)
#     # # axes[0,1].set_xlim(ap[succ_mask].min(), ap[succ_mask].max())

#     # # leave [0,2] empty or use for legend / text
#     # axes[0,2].axis("off")
#     # axes[0,2].text(0.2, 0.5,
#     #             "Top: scatter plots of entropy differences\n"
#     #             "Bottom: distributions of absolute entropies\n"
#     #             "(A, P, D)",
#     #             fontsize=11, va='center')

#     # # --- bottom row: absolute entropy distributions ---
#     # bins = 80
#     # axes[1,0].hist([H_A[succ_mask], H_A[fail_mask]], bins=bins, density=True,
#     #             label=["Success", "Failure"], color=["tab:blue","tab:orange"], alpha=0.6)
#     # axes[1,0].set_title("Entropy of Anchor (H_A)")
#     # axes[1,0].set_xlabel("Entropy")
#     # axes[1,0].legend()
#     # axes[1,0].grid(alpha=0.3)

#     # axes[1,1].hist([H_P[succ_mask], H_P[fail_mask]], bins=bins, density=True,
#     #             label=["Success", "Failure"], color=["tab:blue","tab:orange"], alpha=0.6)
#     # axes[1,1].set_title("Entropy of Paraphrase (H_P)")
#     # axes[1,1].set_xlabel("Entropy")
#     # axes[1,1].legend()
#     # axes[1,1].grid(alpha=0.3)

#     # axes[1,2].hist([H_D[succ_mask], H_D[fail_mask]], bins=bins, density=True,
#     #             label=["Success", "Failure"], color=["tab:blue","tab:orange"], alpha=0.6)
#     # axes[1,2].set_title("Entropy of Distractor (H_D)")
#     # axes[1,2].set_xlabel("Entropy")
#     # axes[1,2].legend()
#     # axes[1,2].grid(alpha=0.3)

#     # plt.suptitle("Entropy Differences and Absolute Entropies by Outcome", fontsize=15, y=1.02)
#     # plt.tight_layout()
#     # plt.savefig("entropy_analysis_grid_apd.png", dpi=1000, bbox_inches="tight")