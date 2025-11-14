
import nethook
import linecache
import torch.nn.functional as F
import torch,os,sys,json
from tqdm import tqdm 
import numpy as np
from helper_functions import _counts, _jaccard_overlap_pct, process_and_write
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path+"/")#add to load modules
from scipy.stats import spearmanr
from visualizations import analyze_and_save_distances
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from model_loaders import get_model
# Tokenize the input texts
from collections import defaultdict


def mean_pool_multi(layer_hidden: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    layer_hidden: [L, B, T, H]
    attention_mask: [B, T] (1=real token, 0=pad)
    returns: [L, B, H] mean over non-pad tokens (prompt included)
    """
    mask = attention_mask.unsqueeze(0).unsqueeze(-1).type_as(layer_hidden)   # [1,B,T,1] -> broadcast
    summed = (layer_hidden * mask).sum(dim=2)                                 # [L,B,H]
    counts = mask.sum(dim=2).clamp(min=1.0)                                   # [L,B,1]
    return summed / counts


def kalm_counterfact_scpp(data_loader,args,layers,device="auto"):
    model = get_model(args.model_type, device=device)
    model.eval()
    # print(model[0].auto_model)
    # for name, mod in model.named_children():
    #     print(name, "->", mod)
    # Example: access specific fields
    # return None
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

    l = layers+["0.auto_model.norm"]
    tr = model[0]
    tokenizer = tr.tokenizer
    with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Rows"):
                try:
                    anchors, paraphrases, distractors, LOS_flags = batch["anchor"], batch["paraphrase"], batch["distractor"], batch["lexical_overlap_flag"]
                    scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
                    scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
                    scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
                        
                    _= model.encode(anchors, convert_to_tensor=True).squeeze(0)
                    batch = tokenizer(
                                anchors,
                                padding=True,
                                return_tensors="pt",
                            ).to(model.device)
                    anchor_sentence_embeddings=torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                    anchor_sentence_embeddings=mean_pool_multi(anchor_sentence_embeddings, batch['attention_mask'])
                    
                    _= model.encode(paraphrases, convert_to_tensor=True).squeeze(0)
                    batch = tokenizer(
                                paraphrases,
                                padding=True,
                                return_tensors="pt",
                            ).to(model.device)
                    paraphrase_sentence_embeddings=torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                    paraphrase_sentence_embeddings=mean_pool_multi(paraphrase_sentence_embeddings, batch['attention_mask'])

                    _= model.encode(distractors, convert_to_tensor=True).squeeze(0)
                    batch = tokenizer(
                                distractors,
                                padding=True,
                                return_tensors="pt",
                            ).to(model.device)
                    distractor_sentence_embeddings=torch.stack([ret[layer_key].output for layer_key in ret],dim=0)
                    distractor_sentence_embeddings=mean_pool_multi(distractor_sentence_embeddings, batch['attention_mask'])
                    print("distractor_sentence_embeddings",distractor_sentence_embeddings.shape)
                except Exception as e:
                    print(paraphrase_sentence_embeddings.shape, batch['attention_mask'].shape)


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
                for i in range(len(anchors)):
                    # print(i,len(anchors))
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
    
    
# def kalm_counterfact_scpp(data_loader,args,device="auto"):
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
#             jaccard_scores_list = []
#             for batch in tqdm(data_loader, desc="Processing Rows"):
#                 # ----- batched fields -----
#                 anchors = batch["anchor"]                  # list[str]
#                 paraphrases = batch["paraphrase"]          # list[str]
#                 distractors = batch["distractor"]          # list[str]
#                 LOS_flags = batch["lexical_overlap_flag"] # list[str]
#                 scores_jaccard = batch.get("score_jaccard", None).tolist()  # optional, list[float]
#                 scores_overlap = batch.get("score_overlap", None).tolist()  # optional, list[float]
#                 scores_containment = batch.get("score_containment", None).tolist()  # optional, list[float]
#                 jaccard_scores_list.extend(scores_jaccard)
#                 anchor_sentence_embeddings= model.encode(anchors, convert_to_tensor=True).squeeze(0)
#                 paraphrase_sentence_embeddings= model.encode(paraphrases, convert_to_tensor=True).squeeze(0)
#                 distractor_sentence_embeddings= model.encode(distractors, convert_to_tensor=True).squeeze(0)

#                 # ----- encode batch (B, D) -----
               
#                 # anchor_sentence_embeddings = model.encode(anchors, convert_to_tensor=True)       # (B, D)
#                 # paraphrase_sentence_embeddings = model.encode(paraphrases, convert_to_tensor=True)# (B, D)
#                 # distractor_sentence_embeddings = model.encode(distractors, convert_to_tensor=True)# (B, D)

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
#                     signed_margins.append(possible_margin_violation)
                  

#                     # correct_pairs.append(distance_anchor_paraphrase)
#                     # pos_pairs.append((anchor, paraphrase))

#                     if(LOS_flag=="low"):
#                         if(failure):
#                             average_margin_violation_lor_low += possible_margin_violation
#                             failure_rate_lor_low += 1
#                         average_margin_lor_low += possible_margin_violation
#                     elif(LOS_flag=="high"):
#                         if(failure):
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


#             rho, pval = spearmanr(jaccard_scores_list,signed_margins)
#             print("Spearman rho:", rho,pval)


#             # optional quantile split
#             jaccard_scores_list = np.array(jaccard_scores_list)   # make sure it's a NumPy array
#             signed_margins = np.array(signed_margins)
#             low_mask  = jaccard_scores_list <= np.quantile(jaccard_scores_list, 0.40)
#             high_mask = jaccard_scores_list >= np.quantile(jaccard_scores_list, 0.60)
#             LOS_sensitivity = signed_margins[high_mask].mean() - signed_margins[low_mask].mean()
#             print("LOS_sensitivity (Q1 vs Q4):", LOS_sensitivity)

#             total_samples = len(data_loader.dataset)
#             total_failures = failure_rate_lor_high + failure_rate_lor_low
#             print("total_failures", total_failures)
#             failure_rate = total_failures / total_samples if total_samples>0 else 0
#             margin_violation = average_margin_violation_lor_high + average_margin_violation_lor_low
#             avg_margin_violation = margin_violation/total_failures if total_failures>0 else 0
#             print("avg_margin_violation", avg_margin_violation)

     
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


def kalm_test_direct_counterfact(file_path,model,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=500
    for name, module in model.named_modules():
        print(name, ":", module.__class__.__name__)

    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    # l=["model.layers.27"]
    with open(file_save_path, 'w') as jsonl_file_writer:
        # with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
            for i in tqdm(range(4000), desc="Processing 500 steps"):
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
                # if violating_pairs:       
                #     print(violating_pairs)     
                # Check comparison
                # if min_local_dist < min_para_dist or max_local_sim > max_para_sim:
                #     print("⚠️ A locality vector is closer to the edit than any paraphrase (Euclidean).",data_entry["edited_prompt"][0],paraphrase_strings[j],locality_strings[i])
                counter+=1
            # break





def kalm_test_scpp(file_path,model,file_save_path,device):
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # l=["encoder.block.6.layer.1.DenseReluDense.wo"]
    # l=["model.layers.27"]
    total_samples=0
    fails=0
    with open(file_save_path, 'w') as jsonl_file_writer:
        # with nethook.TraceDict(model, l) as ret:
        with torch.no_grad():
            # for batch in tqdm(data_loader, desc="Processing batches", leave=False):
           

            data=None
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for obj in tqdm(data, desc="Processing captions"):
                # print(obj)
                # print(obj["caption"],obj["caption2"],obj["negative_caption"])
                caption_tensor = model.encode([obj["caption"]], convert_to_tensor=True,normalize_embeddings=True,).squeeze(0)
                caption2_tensor = model.encode([obj["caption2"]], convert_to_tensor=True,normalize_embeddings=True,).squeeze(0)
                negative_caption_tensor = model.encode([obj["negative_caption"]], convert_to_tensor=True,normalize_embeddings=True,).squeeze(0)

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