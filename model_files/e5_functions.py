import torch.nn.functional as F

from torch import Tensor
from model_loaders import get_model
import torch,os,sys,json
from tqdm import tqdm 
import numpy as np
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path+"/")#add to load modules
from scipy.stats import spearmanr
from visualizations import analyze_and_save_distances
from helper_functions import _jaccard_overlap_pct, _counts
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'



def e5_counterfact_scpp(data_loader,args,device="auto"):
    model,tokenizer = get_model(args.model_name,device)
    model.eval()
    # Example: access specific fields
    
    incorrect_pairs=[]
    correct_pairs=[]
    neg_pairs=[]
    pos_pairs=[]
    average_margin_lor_low=0
    average_margin_violation_lor_low=0
    failure_rate_lor_low=0
    average_margin_lor_high=0
    average_margin_violation_lor_high=0
    failure_rate_lor_high=0
    signed_margins = []
    jaccard_scores_list = []

    # --- collectors for new metrics ---
    all_fail_flags = []          # 1 if triplet fails, else 0
    all_viols = []               # violation magnitude among all (0 if non-failure)
    LO_anchor_paraphrase_list = []              # Jaccard(A,P)
    LO_anchor_distractor_list = []              # Jaccard(A,D)
    LO_paraphrase_distractor_list = []              # Jaccard(P, D)
    LO_negmax_list = []          # max(LO_ad, LO_pd)
    LOc_list = []                # overlap contrast: LO_negmax - LO_ap
    DISTc_list = []              # distance contrast: d_ap - min(d_ad, d_pd)
    file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
    with open(file_save_path, 'w') as jsonl_file_writer:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
    
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]

                batch_dict = tokenizer(anchors, max_length=512, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict.to(device))
                anchor_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                batch_dict = tokenizer(paraphrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict.to(device))
                paraphrase_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                batch_dict = tokenizer(distractors, max_length=512, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict.to(device))
                distractor_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        
                anchor_sentence_embeddings = torch.nn.functional.normalize(anchor_sentence_embeddings, p=2, dim=1)
                paraphrase_sentence_embeddings = torch.nn.functional.normalize(paraphrase_sentence_embeddings, p=2, dim=1)
                distractor_sentence_embeddings = torch.nn.functional.normalize(distractor_sentence_embeddings, p=2, dim=1)
                # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                # --- Cosine similarities ---
                cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embeddings, paraphrase_sentence_embeddings, dim=1)
                cosine_anchor_distractor    = F.cosine_similarity(anchor_sentence_embeddings, distractor_sentence_embeddings, dim=1)
                cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_sentence_embeddings, distractor_sentence_embeddings, dim=1)

                # --- Euclidean distances ---
                distances_anchor_paraphrase = torch.norm(anchor_sentence_embeddings - paraphrase_sentence_embeddings, p=2,dim=1)
                distances_anchor_distractor   = torch.norm(anchor_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=1)
                distances_paraphrase_distractor = torch.norm(paraphrase_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=1)
                    # ----- per-sample pass (preserve your variable names) -----
            # ----- per-sample pass (preserve your variable names) -----
                for i in range(anchor_sentence_embeddings.size(0)):
                    anchor = anchors[i]
                    paraphrase = paraphrases[i]
                    distractor = distractors[i]
                    LOS_flag = LOS_flags[i] # "low" or "high"
                    distance_anchor_paraphrase = distances_anchor_paraphrase[i].item()
                    distance_paraphrase_distractor = distances_paraphrase_distractor[i].item()
                    distance_anchor_distractor = distances_anchor_distractor[i].item()

                    condition_anchor = distance_anchor_distractor < distance_anchor_paraphrase
                    condition_paraphrase = distance_paraphrase_distractor < distance_anchor_paraphrase
                    failure = condition_anchor or condition_paraphrase

                    json_item = {
                        "distance_failure": failure,
                        "lexical_overlap_flag": LOS_flag,
                        "similarity_failure": ((cosine_anchor_distractor[i] > cosine_anchor_paraphrase[i]) or (cosine_paraphrase_distractor[i] > cosine_anchor_paraphrase[i])).item(),
                        "distances": {"dist_cap1_cap2": distance_anchor_paraphrase, "dist_cap1_neg": distance_anchor_distractor, "dist_cap2_neg": distance_paraphrase_distractor},
                        "similarities": {"cos_cap1_cap2": cosine_anchor_paraphrase[i].item(), "cos_cap1_neg": cosine_anchor_distractor[i].item(), "cos_cap2_neg": cosine_paraphrase_distractor[i].item()},
                        "anchor": anchor, "paraphrase": paraphrase, "distractor": distractor,
                        "score_jaccard": scores_jaccard[i], "score_overlap": scores_overlap[i], "score_containment": scores_containment[i]
                    }
                    # print("json_item", json_item)
                    json.dump(json_item, jsonl_file_writer)
                    jsonl_file_writer.write("\n")

                    possible_margin_violation = distance_anchor_paraphrase - min(distance_paraphrase_distractor, distance_anchor_distractor)
                    

                    # --- per-edge lexical overlaps ---
                    LO_ap = _jaccard_overlap_pct(_counts(anchor, "entities"), _counts(paraphrase, "entities"))
                    LO_ad = _jaccard_overlap_pct(_counts(anchor, "entities"), _counts(distractor, "entities"))
                    LO_pd = _jaccard_overlap_pct(_counts(paraphrase, "entities"), _counts(distractor, "entities"))
                    LO_negmax = max(LO_ad, LO_pd)
                    LOc = LO_negmax - LO_ap
                    # DISTc = distance_anchor_paraphrase - min(distance_anchor_distractor, distance_paraphrase_distractor)

                    # --- collect for metrics ---
                    all_fail_flags.append(1 if failure else 0)
                    all_viols.append(max(0.0, -possible_margin_violation))  # violation magnitude (0 if no fail)
                    LO_anchor_paraphrase_list.append(LO_ap)
                    LO_anchor_distractor_list.append(LO_ad)
                    LO_paraphrase_distractor_list.append(LO_pd)
                    LO_negmax_list.append(LO_negmax)
                    LOc_list.append(LOc)
                    DISTc_list.append(possible_margin_violation)

                    # --- keep your existing stratified accumulators ---
                    if (LOS_flag == "low"):
                        if failure:
                            jaccard_scores_list.append(scores_jaccard[i])
                            signed_margins.append(possible_margin_violation)
                            average_margin_violation_lor_low += possible_margin_violation
                            failure_rate_lor_low += 1
                        average_margin_lor_low += possible_margin_violation
                    elif (LOS_flag == "high"):
                        if failure:
                            jaccard_scores_list.append(scores_jaccard[i])
                            signed_margins.append(possible_margin_violation)
                            average_margin_violation_lor_high += possible_margin_violation
                            failure_rate_lor_high += 1
                        average_margin_lor_high += possible_margin_violation

                    if (distance_anchor_distractor <= distance_paraphrase_distractor):
                        incorrect_pairs.append(distance_anchor_distractor)
                        neg_pairs.append((anchor, distractor))
                    else:
                        incorrect_pairs.append(distance_paraphrase_distractor)
                        neg_pairs.append((paraphrase, distractor))

                    correct_pairs.append(distance_anchor_paraphrase)
                    pos_pairs.append((anchor, paraphrase))

            # --- arrays ---
            signed_margins = np.array(signed_margins, dtype=float)
            jaccard_scores_list = np.array(jaccard_scores_list, dtype=float)
            spearman_distc_jaccard, pvalue_distc_jaccard = spearmanr(jaccard_scores_list, signed_margins, nan_policy="omit")

            print("Spearman(DISTc, JaccardOverlap):", spearman_distc_jaccard, "p=", pvalue_distc_jaccard)   

            # =========================
            # Overall Failure Stats
            # =========================
            total_samples = len(data_loader.dataset)
            total_failures = failure_rate_lor_high + failure_rate_lor_low
            failure_rate = (total_failures / total_samples) if total_samples > 0 else 0.0
            margin_violation = average_margin_violation_lor_high + average_margin_violation_lor_low
            avg_margin_violation = (margin_violation / total_failures) if total_failures > 0 else 0.0

            print("total_failures", total_failures)
            print("avg_margin_violation", avg_margin_violation)

            # =========================
            # Fail Rate Stratified by Lexical Overlap
            # =========================
            LO_negmax_list = np.asarray(LO_negmax_list, dtype=float)
            all_fail_flags = np.asarray(all_fail_flags, dtype=int)

            fail_rate_high = np.nan
            fail_rate_low = np.nan
            fail_rate_gap = np.nan
            rel_risk = np.nan

            if LO_negmax_list.size > 0:
                q1, q4 = np.quantile(LO_negmax_list, [0.25, 0.75])
                low_bin = (LO_negmax_list <= q1)
                high_bin = (LO_negmax_list >= q4)

                if low_bin.any():
                    fail_rate_low = float(all_fail_flags[low_bin].mean())
                if high_bin.any():
                    fail_rate_high = float(all_fail_flags[high_bin].mean())
                if low_bin.any() and high_bin.any():
                    fail_rate_gap = float(fail_rate_high - fail_rate_low)
                    rel_risk = float(fail_rate_high / max(fail_rate_low, 1e-12))

            print("FailRate high(Q4) vs low(Q1):", fail_rate_high, fail_rate_low, "Δ=", fail_rate_gap, "RR=", rel_risk)

            # =========================
            # Predictive Power of Overlap
            # =========================
            LOc_list = np.asarray(LOc_list, dtype=float)
            DISTc_list = np.asarray(DISTc_list, dtype=float)
            all_viols = np.asarray(all_viols, dtype=float)

            # AUC: can overlap-contrast predict failure?
            auc_overlap = np.nan
            if all_fail_flags.size > 0 and (all_fail_flags.min() != all_fail_flags.max()):
                try:
                    auc_overlap = float(roc_auc_score(all_fail_flags, LOc_list))
                except Exception:
                    auc_overlap = np.nan
            print("AUC(LO-contrast → failure):", auc_overlap)

            # =========================
            # Failure Severity
            # =========================
            mask_fail = (all_fail_flags == 1)
            sev_gap = np.nan
            if mask_fail.any():
                Jf = LO_negmax_list[mask_fail]
                Vf = all_viols[mask_fail]
                if Jf.size > 0:
                    jf_q1, jf_q4 = np.quantile(Jf, [0.25, 0.75])
                    lo_f = Vf[Jf <= jf_q1]
                    hi_f = Vf[Jf >= jf_q4]
                    if lo_f.size > 0 and hi_f.size > 0:
                        sev_gap = float(np.median(hi_f) - np.median(lo_f))
            print("Failure severity median gap (Q4-Q1 within failures):", sev_gap)

            # --- summary line in JSON ---
            json.dump({
                "Failure Rate": failure_rate,
                "Average Margin Violation": avg_margin_violation,

                # Fail Rate Stratified by Lexical Overlap
                "FailRate_high_Q4": fail_rate_high if not np.isnan(fail_rate_high) else None,
                "FailRate_low_Q1":  fail_rate_low  if not np.isnan(fail_rate_low)  else None,
                "FailRate_gap_Q4_minus_Q1": fail_rate_gap if not np.isnan(fail_rate_gap) else None,
                "RelativeRisk_high_over_low": rel_risk if not np.isnan(rel_risk) else None,

                # Predictive Power of Overlap
                "AUC_LOcontrast_to_failure": auc_overlap if not np.isnan(auc_overlap) else None,

                # Failure Severity
                "Failure_severity_median_gap_Q4_minus_Q1": sev_gap if not np.isnan(sev_gap) else None
            }, jsonl_file_writer)
            jsonl_file_writer.write("\n")



            

    _= analyze_and_save_distances(
        incorrect_pairs,
        correct_pairs,
        title_prefix="Counterfact",
        out_dir=args.save_path,
        # names
        group_neg_name="Non-Paraphrase",
        group_pos_name="Paraphrase",
        # optional ROUGE text pairs
        neg_text_pairs=neg_pairs,
        pos_text_pairs=pos_pairs,
        # >>> NEW: simple styling hooks <<<
        neg_color="#2EA884",    # blue
        pos_color="#D7732D",    # red
        hist_alpha=0.35,
        kde_linewidth=2.0,
        ecdf_linewidth=2.0,
        tau_color="#000000",    # black
        tau_linestyle="--",
        tau_linewidth=1.5,
        violin_facealpha=0.35,
        box_facealpha=0.35,
        )
    




# Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'
# queries = [
#     get_detailed_instruct(task, 'how much protein should a female eat'),
#     get_detailed_instruct(task, '南瓜的家常做法')
# ]
# # No need to add instruction for retrieval documents
# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
# ]
# input_texts = queries + documents


# Tokenize the input texts

# def e5_counterfact_scpp(data_loader, args, device="auto"):
#     import os, re, json
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     from tqdm import tqdm
#     from sklearn.metrics import roc_auc_score
#     from collections import Counter

#     # ---- tiny tokenizer + Jaccard (no external deps) ----
#     _tok = lambda s: set(re.findall(r"\w+", s.lower())) if s else set()
#     def _jacc(a,b):
#         A,B = _tok(a), _tok(b)
#         if not A and not B: return 0.0
#         inter = len(A & B); uni = len(A | B)
#         return inter / uni if uni else 0.0

#     model, tokenizer = get_model(device)
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
#     signed_margins = []
#     jaccard_scores_list = []
#     anchor_lengths_list = []

#     # --- NEW: collectors for discrete metrics (distance-based) ---
#     all_fail_flags = []              # 1/0 per triplet
#     all_viols = []                   # violation magnitude for failures (else 0)
#     LO_ap_list = []                  # Jaccard(A,P)
#     LO_ad_list = []                  # Jaccard(A,D)
#     LO_pd_list = []                  # Jaccard(P,D)
#     LO_negmax_list = []              # max(LO_ad, LO_pd)
#     DISTc_list = []                  # CHANGED: distance contrast: d_ap - min(d_ad, d_pd)
#     LOc_list = []                    # overlap contrast: max(LO_ad,LO_pd) - LO_ap

#     file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with torch.no_grad():
#             for batch in tqdm(data_loader, desc="Processing Rows"):
#                 anchors        = batch["anchor"]
#                 paraphrases    = batch["paraphrase"]
#                 distractors    = batch["distractor"]
#                 LOS_flags      = batch["lexical_overlap_flag"]
#                 scores_jaccard = batch.get("score_jaccard", None).tolist()
#                 scores_overlap = batch.get("score_overlap", None).tolist()
#                 scores_containment = batch.get("score_containment", None).tolist()
#                 anchor_lengths = batch.get("anchor_length", None).tolist()
#                 choices = batch.get("choice", None).tolist()
#                 # ----- encode batch -----
#                 batch_dict = tokenizer(anchors, max_length=512, padding=True, truncation=True, return_tensors='pt')
#                 outputs = model(**batch_dict.to(device))
#                 anchor_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#                 batch_dict = tokenizer(paraphrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
#                 outputs = model(**batch_dict.to(device))
#                 paraphrase_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#                 batch_dict = tokenizer(distractors, max_length=512, padding=True, truncation=True, return_tensors='pt')
#                 outputs = model(**batch_dict.to(device))
#                 distractor_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#                 # ---- L2 normalize (so Euclidean ≈ angular) ----
#                 anchor_sentence_embeddings     = F.normalize(anchor_sentence_embeddings, p=2, dim=1)
#                 paraphrase_sentence_embeddings = F.normalize(paraphrase_sentence_embeddings, p=2, dim=1)
#                 distractor_sentence_embeddings = F.normalize(distractor_sentence_embeddings, p=2, dim=1)

#                 B = anchor_sentence_embeddings.size(0)

#                 for i in range(B):
#                     anchor = anchors[i]; paraphrase = paraphrases[i]; distractor = distractors[i]
#                     LOS_flag = LOS_flags[i]

#                     A = anchor_sentence_embeddings[i]
#                     P = paraphrase_sentence_embeddings[i]
#                     D = distractor_sentence_embeddings[i]

#                     # ---- Euclidean distances (on normalized vectors) ----
#                     d_ap  = torch.norm(A - P, p=2).item()
#                     d_ad  = torch.norm(A - D, p=2).item()
#                     d_pd  = torch.norm(P - D, p=2).item()

#                     # failure (distance view): negative closer than the paraphrase to the anchor pair
#                     condition_anchor     = (d_ad < d_ap)
#                     condition_paraphrase = (d_pd < d_ap)
#                     failure = condition_anchor or condition_paraphrase

#                     # ---- write JSON row (cosine removed; keep field name but distance-based for compatibility) ----
#                     json_item = {
#                         "distance_failure": failure,
#                         "choices": choices[i],
#                         "lexical_overlap_flag": LOS_flag,
#                         "similarity_failure": ((d_ad < d_ap) or (d_pd < d_ap)),  # CHANGED: now distance-based duplicate
#                         "distances": {"dist_cap1_cap2": d_ap, "dist_cap1_neg": d_ad, "dist_cap2_neg": d_pd},
#                         "anchor": anchor, "paraphrase": paraphrase, "distractor": distractor,
#                         "score_jaccard": scores_jaccard[i], "score_overlap": scores_overlap[i], "score_containment": scores_containment[i]
#                     }
#                     json.dump(json_item, jsonl_file_writer); jsonl_file_writer.write("\n")

#                     # signed margin & violation (distance contrast)
#                     m = d_ap - min(d_ad, d_pd)         # >0 means negative is closer than AP (bad)
#                     v = max(0.0, -m)                   # violation magnitude (0 if not failing by margin sign)

#                     # --- lexical overlaps per edge ---
#                     LO_ap = _jacc(anchor, paraphrase)
#                     LO_ad = _jacc(anchor, distractor)
#                     LO_pd = _jacc(paraphrase, distractor)
#                     LO_negmax = max(LO_ad, LO_pd)

#                     # --- contrasts for agreement metric (distance-based) ---
#                     DISTc = d_ap - min(d_ad, d_pd)     # CHANGED: replaces cosine-contrast

#                     # collect for discrete metrics
#                     all_fail_flags.append(1 if failure else 0)
#                     all_viols.append(v)
#                     LO_ap_list.append(LO_ap)
#                     LO_ad_list.append(LO_ad)
#                     LO_pd_list.append(LO_pd)
#                     LO_negmax_list.append(LO_negmax)
#                     DISTc_list.append(DISTc)          # CHANGED
#                     LOc_list.append(LO_negmax - LO_ap)

#                     # keep your existing failure-flagged collections
#                     if (LOS_flag == "low"):
#                         if failure:
#                             anchor_lengths_list.append(anchor_lengths[i])
#                             jaccard_scores_list.append(scores_jaccard[i])
#                             signed_margins.append(m)
#                             average_margin_violation_lor_low += m
#                             failure_rate_lor_low += 1
#                         average_margin_lor_low += m
#                     elif (LOS_flag == "high"):
#                         if failure:
#                             anchor_lengths_list.append(anchor_lengths[i])
#                             jaccard_scores_list.append(scores_jaccard[i])
#                             signed_margins.append(m)
#                             average_margin_violation_lor_high += m
#                             failure_rate_lor_high += 1
#                         average_margin_lor_high += m

#                     # (unchanged bookkeeping for histograms, but distance-based)
#                     if (d_ad <= d_pd):
#                         incorrect_pairs.append(d_ad);   neg_pairs.append((anchor, distractor))
#                     else:
#                         incorrect_pairs.append(d_pd);   neg_pairs.append((paraphrase, distractor))
#                     correct_pairs.append(d_ap);          pos_pairs.append((anchor, paraphrase))

#             # --------- DISCRETE / FAILURE-LOCAL METRICS ----------
#             all_fail_flags = np.asarray(all_fail_flags, dtype=int)
#             all_viols = np.asarray(all_viols, dtype=float)
#             LO_negmax_list = np.asarray(LO_negmax_list, dtype=float)
#             DISTc_list = np.asarray(DISTc_list, dtype=float)
#             LOc_list = np.asarray(LOc_list, dtype=float)

#             # A) Failure Rate Gap (high vs low overlap based on LO_negmax)
#             q1, q4 = np.quantile(LO_negmax_list, [0.25, 0.75])
#             low_bin  = (LO_negmax_list <= q1)
#             high_bin = (LO_negmax_list >= q4)
#             fail_rate_low  = all_fail_flags[low_bin].mean()  if low_bin.any()  else np.nan
#             fail_rate_high = all_fail_flags[high_bin].mean() if high_bin.any() else np.nan
#             fail_rate_gap  = (fail_rate_high - fail_rate_low) if (low_bin.any() and high_bin.any()) else np.nan
#             # relative risk (guard div by zero)
#             rel_risk = (fail_rate_high / max(fail_rate_low, 1e-12)) if (low_bin.any() and high_bin.any()) else np.nan

#             # B) Agreement Rate (within failures): does higher-overlap negative also win in distance?
#             mask_fail = (all_fail_flags == 1)
#             mask_use  = mask_fail & (LOc_list > 0)  # ignore ties where LOc == 0
#             agree = np.mean((DISTc_list[mask_use] > 0).astype(float)) if mask_use.any() else np.nan  # CHANGED

#             # C) AUC of overlap-contrast predicting failure (unchanged target)
#             auc_overlap = np.nan
#             if all_fail_flags.min() != all_fail_flags.max():  # both 0 and 1 present
#                 try:
#                     auc_overlap = roc_auc_score(all_fail_flags, LOc_list)
#                 except Exception:
#                     auc_overlap = np.nan

#             # D) Failure-severity gap (Q4-Q1 of LO_negmax) among failures
#             Jf = LO_negmax_list[mask_fail]
#             Vf = all_viols[mask_fail]
#             sev_gap = np.nan
#             if Jf.size > 0:
#                 jf_q1, jf_q4 = np.quantile(Jf, [0.25, 0.75])
#                 lo_f = Vf[Jf <= jf_q1]
#                 hi_f = Vf[Jf >= jf_q4]
#                 if lo_f.size > 0 and hi_f.size > 0:
#                     sev_gap = float(np.median(hi_f) - np.median(lo_f))

#             # ---------- summary ----------
#             total_samples = len(data_loader.dataset)
#             total_failures = failure_rate_lor_high + failure_rate_lor_low
#             print("total_failures", total_failures)

#             failure_rate = total_failures / total_samples if total_samples > 0 else 0
#             margin_violation = average_margin_violation_lor_high + average_margin_violation_lor_low
#             avg_margin_violation = margin_violation / total_failures if total_failures > 0 else 0
#             print("avg_margin_violation", avg_margin_violation)

#             # --- PRINT the new, discrete metrics ---
#             print("FailRate high(Q4) vs low(Q1):", fail_rate_high, fail_rate_low, "Δ=", fail_rate_gap, "RR=", rel_risk)
#             print("Agreement rate (failures; LOc→DISTc>0):", agree)  # CHANGED label
#             print("AUC(LO-contrast → failure):", auc_overlap)
#             print("Failure severity median gap (Q4-Q1 within failures):", sev_gap)

#             # --- JSON line ---
#             json.dump({
#                 "Failure Rate": failure_rate,
#                 "Average Margin Violation": avg_margin_violation,
#                 "FailRate_high_Q4": float(fail_rate_high) if not np.isnan(fail_rate_high) else None,
#                 "FailRate_low_Q1":  float(fail_rate_low)  if not np.isnan(fail_rate_low)  else None,
#                 "FailRate_gap_Q4_minus_Q1": float(fail_rate_gap) if not np.isnan(fail_rate_gap) else None,
#                 "RelativeRisk_high_over_low": float(rel_risk) if not np.isnan(rel_risk) else None,
#                 "Agreement_rate_failures_DISTANCE": float(agree) if not np.isnan(agree) else None,  # CHANGED key
#                 "AUC_LOcontrast_to_failure": float(auc_overlap) if not np.isnan(auc_overlap) else None,
#                 "Failure_severity_median_gap_Q4_minus_Q1": float(sev_gap) if not np.isnan(sev_gap) else None
#             }, jsonl_file_writer)
#             jsonl_file_writer.write("\n")
# def e5_counterfact_scpp(data_loader, args, device="auto"):


#     # ---- tiny tokenizer + Jaccard (no external deps) ----
    

#     model, tokenizer = get_model(device)
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
#     signed_margins = []
#     jaccard_scores_list = []

#     # --- collectors for new metrics ---
#     all_fail_flags = []          # 1 if triplet fails, else 0
#     all_viols = []               # violation magnitude among all (0 if non-failure)
#     LO_anchor_paraphrase_list = []              # Jaccard(A,P)
#     LO_anchor_distractor_list = []              # Jaccard(A,D)
#     LO_paraphrase_distractor_list = []              # Jaccard(P, D)
#     LO_negmax_list = []          # max(LO_ad, LO_pd)
#     LOc_list = []                # overlap contrast: LO_negmax - LO_ap
#     DISTc_list = []              # distance contrast: d_ap - min(d_ad, d_pd)

#     file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
#     with open(file_save_path, 'w') as jsonl_file_writer:
#         with torch.no_grad():
#             for batch in tqdm(data_loader, desc="Processing Rows"):
#                 # ----- batched fields -----
#                 anchors = batch["anchor"]                   # list[str]
#                 paraphrases = batch["paraphrase"]           # list[str]
#                 distractors = batch["distractor"]           # list[str]
#                 LOS_flags = batch["lexical_overlap_flag"]   # list[str]
#                 scores_jaccard = batch.get("score_jaccard", None).tolist()        # optional, list[float]
#                 scores_overlap = batch.get("score_overlap", None).tolist()        # optional, list[float]
#                 scores_containment = batch.get("score_containment", None).tolist()# optional, list[float]

#                 # ----- encode batch (B, D) -----
#                 batch_dict = tokenizer(anchors, max_length=512, padding=True, truncation=True, return_tensors='pt')
#                 outputs = model(**batch_dict.to(device))
#                 anchor_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#                 batch_dict = tokenizer(paraphrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
#                 outputs = model(**batch_dict.to(device))
#                 paraphrase_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#                 batch_dict = tokenizer(distractors, max_length=512, padding=True, truncation=True, return_tensors='pt')
#                 outputs = model(**batch_dict.to(device))
#                 distractor_sentence_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#                 # ----- normalize row-wise -----
#                 anchor_sentence_embeddings = F.normalize(anchor_sentence_embeddings, p=2, dim=1)
#                 paraphrase_sentence_embeddings = F.normalize(paraphrase_sentence_embeddings, p=2, dim=1)
#                 distractor_sentence_embeddings = F.normalize(distractor_sentence_embeddings, p=2, dim=1)

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

#                     json_item = {
#                         "distance_failure": failure,
#                         "lexical_overlap_flag": LOS_flag,
#                         "similarity_failure": (cosine_anchor_distractor > cosine_anchor_paraphrase) or (cosine_paraphrase_distractor > cosine_anchor_paraphrase),
#                         "distances": {"dist_cap1_cap2": distance_anchor_paraphrase, "dist_cap1_neg": distance_anchor_distractor, "dist_cap2_neg": distance_paraphrase_distractor},
#                         "similarities": {"cos_cap1_cap2": cosine_anchor_paraphrase, "cos_cap1_neg": cosine_anchor_distractor, "cos_cap2_neg": cosine_paraphrase_distractor},
#                         "anchor": anchor, "paraphrase": paraphrase, "distractor": distractor,
#                         "score_jaccard": scores_jaccard[i], "score_overlap": scores_overlap[i], "score_containment": scores_containment[i]
#                     }
#                     json.dump(json_item, jsonl_file_writer)
#                     jsonl_file_writer.write("\n")

#                     possible_margin_violation = distance_anchor_paraphrase - min(distance_paraphrase_distractor, distance_anchor_distractor)
#                     signed_margins.append(possible_margin_violation)

#                     # --- per-edge lexical overlaps ---
#                     LO_ap = _jaccard_overlap_pct(anchor, paraphrase)
#                     LO_ad = _jaccard_overlap_pct(anchor, distractor)
#                     LO_pd = _jaccard_overlap_pct(paraphrase, distractor)
#                     LO_negmax = max(LO_ad, LO_pd)
#                     LOc = LO_negmax - LO_ap
#                     DISTc = distance_anchor_paraphrase - min(distance_anchor_distractor, distance_paraphrase_distractor)

#                     # --- collect for metrics ---
#                     all_fail_flags.append(1 if failure else 0)
#                     all_viols.append(max(0.0, -possible_margin_violation))  # violation magnitude (0 if no fail)
#                     LO_anchor_paraphrase_list.append(LO_ap)
#                     LO_anchor_distractor_list.append(LO_ad)
#                     LO_paraphrase_distractor_list.append(LO_pd)
#                     LO_negmax_list.append(LO_negmax)
#                     LOc_list.append(LOc)
#                     DISTc_list.append(DISTc)

#                     # --- keep your existing stratified accumulators ---
#                     if (LOS_flag == "low"):
#                         if failure:
#                             jaccard_scores_list.append(scores_jaccard[i])
#                             average_margin_violation_lor_low += possible_margin_violation
#                             failure_rate_lor_low += 1
#                         average_margin_lor_low += possible_margin_violation
#                     elif (LOS_flag == "high"):
#                         if failure:
#                             jaccard_scores_list.append(scores_jaccard[i])
#                             average_margin_violation_lor_high += possible_margin_violation
#                             failure_rate_lor_high += 1
#                         average_margin_lor_high += possible_margin_violation

#                     if (distance_anchor_distractor <= distance_paraphrase_distractor):
#                         incorrect_pairs.append(distance_anchor_distractor)
#                         neg_pairs.append((anchor, distractor))
#                     else:
#                         incorrect_pairs.append(distance_paraphrase_distractor)
#                         neg_pairs.append((paraphrase, distractor))

#                     correct_pairs.append(distance_anchor_paraphrase)
#                     pos_pairs.append((anchor, paraphrase))

#             # --- arrays ---
#             signed_margins = np.array(signed_margins, dtype=float)
#             jaccard_scores_list = np.array(jaccard_scores_list, dtype=float)

#             # =========================
#             # Overall Failure Stats
#             # =========================
#             total_samples = len(data_loader.dataset)
#             total_failures = failure_rate_lor_high + failure_rate_lor_low
#             failure_rate = (total_failures / total_samples) if total_samples > 0 else 0.0
#             margin_violation = average_margin_violation_lor_high + average_margin_violation_lor_low
#             avg_margin_violation = (margin_violation / total_failures) if total_failures > 0 else 0.0

#             print("total_failures", total_failures)
#             print("avg_margin_violation", avg_margin_violation)

#             # =========================
#             # Fail Rate Stratified by Lexical Overlap
#             # =========================
#             LO_negmax_list = np.asarray(LO_negmax_list, dtype=float)
#             all_fail_flags = np.asarray(all_fail_flags, dtype=int)

#             fail_rate_high = np.nan
#             fail_rate_low = np.nan
#             fail_rate_gap = np.nan
#             rel_risk = np.nan

#             if LO_negmax_list.size > 0:
#                 q1, q4 = np.quantile(LO_negmax_list, [0.25, 0.75])
#                 low_bin = (LO_negmax_list <= q1)
#                 high_bin = (LO_negmax_list >= q4)

#                 if low_bin.any():
#                     fail_rate_low = float(all_fail_flags[low_bin].mean())
#                 if high_bin.any():
#                     fail_rate_high = float(all_fail_flags[high_bin].mean())
#                 if low_bin.any() and high_bin.any():
#                     fail_rate_gap = float(fail_rate_high - fail_rate_low)
#                     rel_risk = float(fail_rate_high / max(fail_rate_low, 1e-12))

#             print("FailRate high(Q4) vs low(Q1):", fail_rate_high, fail_rate_low, "Δ=", fail_rate_gap, "RR=", rel_risk)

#             # =========================
#             # Predictive Power of Overlap
#             # =========================
#             LOc_list = np.asarray(LOc_list, dtype=float)
#             DISTc_list = np.asarray(DISTc_list, dtype=float)
#             all_viols = np.asarray(all_viols, dtype=float)

#             # AUC: can overlap-contrast predict failure?
#             auc_overlap = np.nan
#             if all_fail_flags.size > 0 and (all_fail_flags.min() != all_fail_flags.max()):
#                 try:
#                     auc_overlap = float(roc_auc_score(all_fail_flags, LOc_list))
#                 except Exception:
#                     auc_overlap = np.nan
#             print("AUC(LO-contrast → failure):", auc_overlap)

#             # =========================
#             # Failure Severity
#             # =========================
#             mask_fail = (all_fail_flags == 1)
#             sev_gap = np.nan
#             if mask_fail.any():
#                 Jf = LO_negmax_list[mask_fail]
#                 Vf = all_viols[mask_fail]
#                 if Jf.size > 0:
#                     jf_q1, jf_q4 = np.quantile(Jf, [0.25, 0.75])
#                     lo_f = Vf[Jf <= jf_q1]
#                     hi_f = Vf[Jf >= jf_q4]
#                     if lo_f.size > 0 and hi_f.size > 0:
#                         sev_gap = float(np.median(hi_f) - np.median(lo_f))
#             print("Failure severity median gap (Q4-Q1 within failures):", sev_gap)

#             # --- summary line in JSON ---
#             json.dump({
#                 "Failure Rate": failure_rate,
#                 "Average Margin Violation": avg_margin_violation,

#                 # Fail Rate Stratified by Lexical Overlap
#                 "FailRate_high_Q4": fail_rate_high if not np.isnan(fail_rate_high) else None,
#                 "FailRate_low_Q1":  fail_rate_low  if not np.isnan(fail_rate_low)  else None,
#                 "FailRate_gap_Q4_minus_Q1": fail_rate_gap if not np.isnan(fail_rate_gap) else None,
#                 "RelativeRisk_high_over_low": rel_risk if not np.isnan(rel_risk) else None,

#                 # Predictive Power of Overlap
#                 "AUC_LOcontrast_to_failure": auc_overlap if not np.isnan(auc_overlap) else None,

#                 # Failure Severity
#                 "Failure_severity_median_gap_Q4_minus_Q1": sev_gap if not np.isnan(sev_gap) else None
#             }, jsonl_file_writer)
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
    

def e5_test_scpp(file_path,model,tokenizer,file_save_path,device):
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
                input_texts=[obj["caption"],obj["caption2"],obj["negative_caption"]]
                batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

                outputs = model(**batch_dict.to(device))
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                # print(embeddings.shape)
                caption_tensor=embeddings[0]
                caption2_tensor=embeddings[1]
                negative_caption_tensor=embeddings[2]
                # caption_tensor = model.encode([obj["caption"]], convert_to_tensor=True).squeeze(0)
                # caption2_tensor = model.encode([obj["caption2"]], convert_to_tensor=True).squeeze(0)
                # negative_caption_tensor = model.encode([obj["negative_caption"]], convert_to_tensor=True).squeeze(0)

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
        json.dump({"Accuracy": (total_samples - fails) / total_samples if total_samples > 0 else 0 }, jsonl_file_writer)
        jsonl_file_writer.write("\n")