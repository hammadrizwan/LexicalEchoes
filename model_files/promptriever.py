from sentence_transformers import SentenceTransformer
import nethook
import linecache
import torch.nn.functional as F
import torch,os,sys,json
from tqdm import tqdm 
import numpy as np
from helper_functions import _counts, _jaccard_overlap_pct
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path+"/")#add to load modules
from scipy.stats import spearmanr
from visualization_quora_paws import analyze_and_save_distances
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel
class Promptriever:
    def __init__(self, model_name_or_path,device="cuda"):
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        self.model.eval().to(device)

    def get_model(self, peft_model_name):
        # Load the PEFT configuration to get the base model name
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Load the base model and tokenizer
        base_model = AutoModel.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Load and merge the PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()

        # can be much longer, but for the example 512 is enough
        model.config.max_length = 512
        tokenizer.model_max_length = 512

        return model, tokenizer

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(self, sentences, max_length: int = 2048, batch_size: int = 4):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def load_model(device="auto"):
    model = Promptriever("samaya-ai/promptriever-llama3.1-8b-instruct-v1",device)
    return model
# Tokenize the input texts




def promptretriever_counterfact_scpp(data_loader,args,device="auto"):
    model = load_model(device)
    print(next(model.model.parameters()).device)
    # model.eval()
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
    instruction = "A relevant document would express the same meaning and intent as the query without adding, removing, or altering information. Think carefully about these conditions when determining relevance."
    
    file_save_path = os.path.join(args.save_path, "counterfact_results.jsonl")
    with open(file_save_path, 'w') as jsonl_file_writer:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
    
                anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]


                anchors_instruction = [f"query: {q.strip()} {instruction.strip()}".strip() for q in anchors]
                paraphrases_instruction = [f"query: {q.strip()} {instruction.strip()}".strip() for q in paraphrases]
                paraphrases_passage =["passage "+paraphrase for paraphrase in paraphrases]
                distractors_passage =["passage "+distractor for distractor in distractors]
                

                #normalization is internal

                anchor_sentence_embeddings = torch.tensor(model.encode(anchors_instruction))
                paraphrase_prompt_sentence_embeddings = torch.tensor(model.encode(paraphrases_instruction))
                paraphrase_sentence_embeddings = torch.tensor(model.encode(paraphrases_passage))
                distractor_sentence_embeddings = torch.tensor(model.encode(distractors_passage))
                print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)


        
                
                # print("anchor_sentence_embeddings",anchor_sentence_embeddings.shape)
                # --- Cosine similarities ---
                cosine_anchor_paraphrase   = F.cosine_similarity(anchor_sentence_embeddings, paraphrase_sentence_embeddings, dim=1)
                cosine_anchor_distractor    = F.cosine_similarity(anchor_sentence_embeddings, distractor_sentence_embeddings, dim=1)
                cosine_paraphrase_distractor = F.cosine_similarity(paraphrase_prompt_sentence_embeddings, distractor_sentence_embeddings, dim=1)

                # --- Euclidean distances ---
                distances_anchor_paraphrase = torch.norm(anchor_sentence_embeddings - paraphrase_sentence_embeddings, p=2,dim=1)
                distances_anchor_distractor   = torch.norm(anchor_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=1)
                distances_paraphrase_distractor = torch.norm(paraphrase_prompt_sentence_embeddings - distractor_sentence_embeddings, p=2,dim=1)

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