import argparse
import sys
from xml.parsers.expat import model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from transformers.utils import default_cache_path
# from LexicalBias.Lexical_Semantic_Quantification.quora_paws import llama_it_quora_paws
import nethook
import os,json,sys
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple, Dict
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from model_files.helper_functions import run_all,_tokens_no_entities,split_by_median
from model_files.qwen import qwen_counterfact_scpp
from model_files.gemma_functions import gemma_pt_counterfact_scpp,gemma_it_counterfact_scpp,gemma_counterfact_dime
from model_files.llama_functions import llama_it_counterfact_scpp,llama_pt_counterfact_scpp
from model_files.e5_functions import e5_counterfact_scpp
from model_files.kalm_functions import kalm_counterfact_scpp
from model_files.promptriever import promptretriever_counterfact_scpp
import statistics
# from llama_functions import llama_embeddings_analysis_counterfact_average,llama_embeddings_analysis_counterfact_lasttoken
# from model_files.gemma_functions import gemma_test_direct_counterfact_easyedit
from get_token import *

LAYER_TEMPLATE_DICT={"google/gemma-3-1b-pt":["model.layers.{}"],"google/gemma-3-4b-pt":["model.language_model.layers.{}"],"google/gemma-3-12b-pt":["model.language_model.layers.{}"],
                    "google/gemma-3-1b-it":["model.layers.{}"],"google/gemma-3-4b-it":["model.language_model.layers.{}"],"google/gemma-3-12b-it":["model.language_model.layers.{}"],
                    "meta-llama/Llama-3.2-1B":["model.layers.{}"],"meta-llama/Llama-3.2-1B-Instruct":["model.layers.{}"],"meta-llama/Llama-3.2-3B":["model.layers.{}"],"meta-llama/Llama-3.2-3B-Instruct":["model.layers.{}"],
                    "Qwen/Qwen3-Embedding-8B":["layers.{}"],"HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5":["0.auto_model.layers.{}"]#add the other models and fix qwen
                    }
MODEL_MAPPING_DICT={"qwen":"Qwen/Qwen3-Embedding-8B","kalm":"HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5","e5":"intfloat/multilingual-e5-large-instruct",
                    "promptriever":"samaya-ai/promptriever-llama3.1-8b-instruct-v1"}

LAYER_MAPPING_DICT={"meta-llama/Llama-3.2-3B-Instruct":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                    "meta-llama/Llama-3.2-3B":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                    "meta-llama/Llama-3.2-1B-Instruct":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                    "meta-llama/Llama-3.2-1B":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                    "google/gemma-3-1b-pt":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                    "google/gemma-3-1b-it":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                    "google/gemma-3-4b-pt":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],
                    "google/gemma-3-4b-it":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],
                    "google/gemma-3-12b-pt":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
                    "google/gemma-3-12b-it":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
                    "Qwen/Qwen3-Embedding-8B":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
                    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
                    }
                    

ACCESS_TOKEN=get_token()


#----------------------------------------------------------------------------
# Section: Counterfact Dataset from EasyEdit
#----------------------------------------------------------------------------
# region CounterFactDatasetEasyEdit

# def robust_quantile_threshold(scores, target_high_frac=0.5):
#     """
#     Returns a threshold such that roughly target_high_frac of samples become 'high'.
#     Works even when many values are identical (low variance).
#     """
#     n = len(scores)
#     if n == 0:
#         return 0.0
#     scores_sorted = sorted(scores)
#     # index where 'high' begins (top target_high_frac)
#     # e.g., target_high_frac=0.5 -> split in half
#     split_at = max(0, min(n - 1, int(round((1.0 - target_high_frac) * (n - 1)))))
#     return scores_sorted[split_at]

class CounterFactDatasetEasyEdit(Dataset):
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            dataset = json.load(f)   # data is a list of dicts
        tmp_rows=[]
        score_jaccard_list=[]
        score_containment_list=[]
        score_overlap_list=[]
        for record in dataset:
            anchor=record["prompt"]
            paraphrase=record["rephrase_prompt"]
            distractor=record["locality_prompt"]
            rec={"anchor":anchor,"paraphrase":paraphrase,"distractor":distractor}
            score_dict=run_all(anchor,paraphrase, distractor)
            score_jaccard=score_dict["jaccard"]
            score_overlap=score_dict["overlap_coeff"]
            score_containment=score_dict["containment"]
            score_jaccard_list.append(score_jaccard)
            score_overlap_list.append(score_overlap)
            score_containment_list.append(score_containment)

            # overlaps.append(pct_pd)

            tmp_rows.append({
                **rec,  # keep original fields if you want to return them too
                "anchor": anchor,
                "paraphrase": paraphrase,
                "distractor": distractor,
                "score_jaccard": score_jaccard,
                "score_containment": score_containment,
                "score_overlap": score_overlap,
            })

        # Compute "middle ground" thresholds (medians)
        # median = statistics.median(overlaps) if overlaps else 0.0
        # threshold=round(median, 2)
        type_threshold="score_jaccard"
        if(type_threshold=="score_jaccard"):
            threshold, labels = split_by_median(score_jaccard_list)
        elif(type_threshold=="score_containment"):
            threshold, labels = split_by_median(score_containment_list)
        else:#type_threshold=="score_overlap"
            threshold, labels = split_by_median(score_overlap_list)

        # Second pass: assign flags using medians
        self.data = []
        self.count_low_flags=0
        self.count_high_flags=0
        for row in tmp_rows:
            if(row[type_threshold] > threshold):
                flag="high"
                self.count_high_flags+=1
            else:
                flag="low"
                self.count_low_flags+=1
            self.data.append({
                **row,
                "lexical_overlap_flag": flag
            })
        print("self.count_high_flags",self.count_high_flags)
        print("self.count_low_flags",self.count_low_flags)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]#data already in required format
        return item
# endregion


#----------------------------------------------------------------------------
# Section: Counterfact Dataset from PENME
#----------------------------------------------------------------------------
# region CounterFactDatasetPenme
class CounterFactDatasetPenme(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as jsonl_file:
          lines = jsonl_file.readlines()
        
        dataset = [json.loads(line) for line in lines]
        
        tmp_rows=[]
        score_jaccard_list=[]
        score_containment_list=[]
        score_overlap_list=[]
        for row in dataset[:1000]:
            anchor=row["edited_prompt"][0]
            distractor_list=[]
            distractor_list.extend(row["neighborhood_prompts_high_sim"])
            distractor_list.extend(row["neighborhood_prompts_low_sim"])
            distractor,best_scores=self.get_max_overlap(anchor,distractor_list)
            # distractor= random.choice(distractor_list)

            paraphrase=row["edited_prompt_paraphrases_processed_testing"]
            rec={"anchor":anchor,"paraphrase":paraphrase,"distractor":distractor}
            score_dict=run_all(anchor,paraphrase, distractor,"with_entities")
            score_jaccard=score_dict["jaccard"]
            score_overlap=score_dict["overlap_coeff"]
            score_containment=score_dict["containment"]
            score_jaccard_list.append(score_jaccard)
            score_overlap_list.append(score_overlap)
            score_containment_list.append(score_containment)

            tmp_rows.append({
                **rec,  # keep original fields if you want to return them too
                "anchor": anchor,
                # "anchor_length": best_scores["query_len"],
                "paraphrase": paraphrase,
                "distractor": distractor,
                "score_jaccard": score_jaccard,
                "score_containment": score_containment,
                "score_overlap": score_overlap,
          
              
            })
        # print("score_jaccard_list",score_jaccard_list[:10])
        type_threshold="score_jaccard"
        if(type_threshold=="score_jaccard"):
            threshold, labels = split_by_median(score_jaccard_list)
            dist = Counter(score_jaccard_list)
        elif(type_threshold=="score_containment"):
            threshold, labels = split_by_median(score_containment_list)
            dist = Counter(score_containment_list)
        else:#type_threshold=="score_overlap"
            threshold, labels = split_by_median(score_overlap_list)
            dist = Counter(score_overlap_list)

        print("Exact score distribution:")
        for score, count in dist.most_common():  # sorted by frequency
            print(f"Score {score:.2f} : {count} samples")

 
        print(f"Chosen threshold for 'high' overlap: {threshold}")
        # threshold=10.0
        # Second pass: assign flags using medians
        self.data = []
        self.count_low_flags=0
        self.count_high_flags=0
        for row in tmp_rows:
            if(row[type_threshold] > threshold):
                flag="high"
                self.count_high_flags+=1
            else:
                flag="low"
                self.count_low_flags+=1
            self.data.append({
                **row,
                "lexical_overlap_flag": flag
            })
        print("self.count_high_flags",self.count_high_flags)
        print("self.count_low_flags",self.count_low_flags)
        # self.data=self.data[:2]#debug
    def __len__(self):
        return len(self.data)
    def get_min_overlap(
            self,
            sentence: str,
            distractor_list: List[str]
        ) -> Tuple[Optional[str], Dict[str, float]]:
        query_tokens = Counter(_tokens_no_entities(sentence))
        best: Optional[str] = None
        # (overlap_count, jaccard) — we now minimize this tuple
        best_key = (math.inf, math.inf)
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
            if key < best_key:  # minimize overlap, then Jaccard
                best_key = key
                best = cand
                best_scores = {
                    "overlap_count": overlap_count,
                    "jaccard": jaccard,
                    "candidate_len": sum(cand_tokens.values()),
                    "query_len": sum(query_tokens.values()),
                }

        return best, best_scores

    def get_max_overlap(
        self,
        sentence: str,
        distractor_list: List[str]
    ) -> Tuple[Optional[str], Dict[str, float]]:
  

        

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

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
    
# endregion
    


#----------------------------------------------------------------------------
# Section: Argument Parsing
#----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Lexical Bias Benchmarking Counterfact EasyEdit")

    # Data and paths
    parser.add_argument(
        "--mode",
        type=bool,
        default=False,
        choices=[True,False],
        help="Averaging or Last Token (True for averaging, False for last token)"
    )
    parser.add_argument(
        "--dataset_path_easyedit",
        type=str,
        default="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/counterfact/counterfact_easyedit/counterfact-train.json",
        help="Path to the dataset file or directory EasyEdit"
    )
    parser.add_argument(
        "--dataset_path_penme",
        type=str,
        default="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/counterfact/Counterfact_OpenAI.jsonl",
        help="Path to the dataset file or directory PENME"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="penme",
        choices=["penme","easyedit"],
        help="Data source type for counterfact (penme or easyedit)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./counterfact_analysis/",
        help="Path to save results (jsonl, csv, etc.)"
    )

    # Model settings
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen",
        choices=["kalm","e5","qwen","promptriever"
                 , "google/gemma-3-1b-pt","google/gemma-3-4b-pt","google/gemma-3-12b-pt","google/gemma-3-1b-it","google/gemma-3-4b-it","google/gemma-3-12b-it",
                 "meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B","meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct"],
        help="Type of model to use for embeddings"
    )

    # Evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu","auto"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--whitening",
        type=bool,
        default=False,
        choices=[True, False],
        help="Random seed for reproducibility"
    )
    parser.add_argument(#change to dictionary
        "--whitening_stats_path",
        type=str,
        default="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/common_pile_analysis/whitening_stats.pt",
        help="Path to whitening stats (if whitening is True)"
    )
   
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    return args
#----------------------------------------------------------------------------



if __name__ == "__main__":
    args = get_args()
    print("args",args.mode)
    if(args.device=="auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    torch.manual_seed(args.seed)
    if(args.data_type=="easyedit"):
        file_path_counterfact=args.dataset_path_easyedit
        dataset= CounterFactDatasetEasyEdit(args.dataset_path_easyedit)
    else:
        file_path_counterfact=args.dataset_path_penme
        dataset= CounterFactDatasetPenme(args.dataset_path_penme)
    if("gemma" in args.model_type or "Llama" in args.model_type):
        if(str(args.mode)=="True"):
            args.save_path=args.save_path+args.model_type+"_averaging/"
        else:
            args.save_path=args.save_path+args.model_type+"_lasttoken/"
    else:
        args.save_path=args.save_path+args.model_type+"/"
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if("Llama" in args.model_type and "Instruct" in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        formated_layers=[]
        for layer in LAYER_MAPPING_DICT[args.model_type]:
            layer_string=LAYER_TEMPLATE_DICT[args.model_type][0].format(layer)
            print(layer_string)
            layer_file_path=args.save_path+str(layer_string)+"/"
            if not os.path.exists(layer_file_path):
                os.makedirs(layer_file_path)
            formated_layers.append(layer_string)

        llama_it_counterfact_scpp(data_loader,args,ACCESS_TOKEN,formated_layers,device)
    elif("Llama" in args.model_type and "Instruct" not in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        formated_layers=[]
        for layer in LAYER_MAPPING_DICT[args.model_type]:
            layer_string=LAYER_TEMPLATE_DICT[args.model_type][0].format(layer)
            print(layer_string)
            layer_file_path=args.save_path+str(layer_string)+"/"
            if not os.path.exists(layer_file_path):
                os.makedirs(layer_file_path)
            formated_layers.append(layer_string)
        llama_pt_counterfact_scpp(data_loader,args,ACCESS_TOKEN,formated_layers,device)
    elif("gemma" in args.model_type and "pt" in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        formated_layers=[]
        for layer in LAYER_MAPPING_DICT[args.model_type]:
            layer_string=LAYER_TEMPLATE_DICT[args.model_type][0].format(layer)
            print(layer_string)
            layer_file_path=args.save_path+str(layer_string)+"/"
            if not os.path.exists(layer_file_path):
                os.makedirs(layer_file_path)
            formated_layers.append(layer_string)
        # gemma_counterfact_dime(data_loader,args,ACCESS_TOKEN,formated_layers,device)
        gemma_pt_counterfact_scpp(data_loader,args,ACCESS_TOKEN,formated_layers,device)
        # gemma_pt_counterfact_scpp(data_loader,args,ACCESS_TOKEN,LAYER_MAPPING_DICT,device)
    elif("gemma" in args.model_type and "it" in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        formated_layers=[]
        for layer in LAYER_MAPPING_DICT[args.model_type]:
            layer_string=LAYER_TEMPLATE_DICT[args.model_type][0].format(layer)
            print(layer_string)
            layer_file_path=args.save_path+str(layer_string)+"/"
            if not os.path.exists(layer_file_path):
                os.makedirs(layer_file_path)
            formated_layers.append(layer_string)

        gemma_it_counterfact_scpp(data_loader,args,ACCESS_TOKEN,formated_layers,device)
        # gemma_it_counterfact_scpp(data_loader,args,ACCESS_TOKEN,LAYER_MAPPING_DICT,device)
    elif(args.model_type=="qwen"):
        args.model_type = MODEL_MAPPING_DICT[args.model_type]
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        formated_layers=[]
        for layer in LAYER_MAPPING_DICT[args.model_type]:
            layer_string=LAYER_TEMPLATE_DICT[args.model_type][0].format(layer)
            print(layer_string)
            layer_file_path=args.save_path+str(layer_string)+"/"
            if not os.path.exists(layer_file_path):
                os.makedirs(layer_file_path)
            formated_layers.append(layer_string)
        qwen_counterfact_scpp(data_loader,args,formated_layers,device)
    elif(args.model_type=="e5"):
        args.model_type = MODEL_MAPPING_DICT[args.model_type]
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        e5_counterfact_scpp(data_loader,args,device)
    elif(args.model_type=="kalm"):
        args.model_type = MODEL_MAPPING_DICT[args.model_type]
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        formated_layers=[]
        for layer in LAYER_MAPPING_DICT[args.model_type]:
            layer_string=LAYER_TEMPLATE_DICT[args.model_type][0].format(layer)
            print(layer_string)
            layer_file_path=args.save_path+str(layer_string)+"/"
            if not os.path.exists(layer_file_path):
                os.makedirs(layer_file_path)
            formated_layers.append(layer_string)
        kalm_counterfact_scpp(data_loader,args,formated_layers,device)
    elif(args.model_type=="promptriever"):
        args.model_type = MODEL_MAPPING_DICT[args.model_type]
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        promptretriever_counterfact_scpp(data_loader,args,device)
      
    
    
    # gemma(device)
    # qwen(device)
    # print(default_cache_path)
    # hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    # print("HF_HOME =", os.getenv("HF_HOME"))
    # print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
    # print("HF_DATASETS_CACHE =", os.getenv("HF_DATASETS_CACHE"))
    # print("Default HF cache path (transformers):", default_cache_path)
    # print("Datasets cache (datasets):", hf_datasets_cache)