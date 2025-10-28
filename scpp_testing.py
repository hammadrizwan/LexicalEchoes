from xml.parsers.expat import model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import nethook
import os,json, gc, argparse, torch
dir_path = os.path.dirname(os.path.abspath(__file__))
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import linecache
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from model_files.qwen import qwen_counterfact_scpp
from model_files.gemma_functions import gemma_pt_counterfact_scpp,gemma_it_counterfact_scpp,gemma_counterfact_dime
from model_files.llama_functions import llama_it_counterfact_scpp,llama_pt_counterfact_scpp
from model_files.helper_functions import run_all,_tokens_no_entities,split_by_median
from model_files.e5_functions import e5_counterfact_scpp
from model_files.kalm_functions import kalm_counterfact_scpp
from model_files.promptriever import promptretriever_counterfact_scpp
from get_token import *

LAYER_TEMPLATE_DICT={"gemma-3-1b-pt":["model.layers.{}"],"gemma-3-4b-pt":["model.language_model.layers.{}"],"gemma-3-12b-pt":["model.language_model.layers.{}"],
                    "gemma-3-1b-it":["model.layers.{}"],"gemma-3-4b-it":["model.language_model.layers.{}"],"gemma-3-12b-it":["model.language_model.layers.{}"],
                    "Llama-3.2-1B":["model.layers.{}"],"Llama-3.2-1B-Instruct":["model.layers.{}"],"Llama-3.2-3B":["model.layers.{}"],"Llama-3.2-3B-Instruct":["model.layers.{}"],
                    "qwen":["layers.{}"]
                    }

LAYER_MAPPING_DICT={"Llama-3.2-3B-Instruct":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                    "Llama-3.2-3B":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                    "Llama-3.2-1B-Instruct":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                    "Llama-3.2-1B":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                    "gemma-3-1b-pt":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                    "gemma-3-1b-it":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                    "gemma-3-4b-pt":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],
                    "gemma-3-4b-it":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],
                    "gemma-3-12b-pt":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
                    "gemma-3-12b-it":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
                    "qwen":[33,34,35]
                    }

ACCESS_TOKEN=get_token()

LOAD_PATH="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/scpp/data/"

#----------------------------------------------------------------------------
# Section: Argument Parsing
#----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Lexical Bias Benchmarking")
    parser.add_argument(
        "--mode",
        type=bool,
        default=False,
        choices=[True,False],
        help="Averaging or Last Token (True for averaging, False for last token)"
    )
    # Data and paths
    parser.add_argument(
        "--data_type",
        type=str,
        default="auto",
        choices=["replace_att","replace_obj","swap_att","swap_obj","replace_rel","auto"],#auto is run all
        help="Path to the dataset file or directory"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./scpp_analysis/",
        help="Path to save results (jsonl, csv, etc.)"
    )

    # Model settings
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen",
        choices=["qwen", "llama","kalm","e5","promptriever",
                  "gemma-3-1b-pt","gemma-3-4b-pt","gemma-3-12b-pt","gemma-3-1b-it","gemma-3-4b-it","gemma-3-12b-it",
                 "Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"],
        help="Type of model to use for embeddings"
    )

    # Training / evaluation settings
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




# Overlap Coefficient (a.k.a. Szymkiewicz–Simpson)
def overlap_coeff_pct(a: str, b: str) -> float:
    A = Counter(w.lower() for w in word_tokenize(a)) if a else Counter()
    B = Counter(w.lower() for w in word_tokenize(b)) if b else Counter()
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = min(len(A), len(B))
    return 100.0 * inter / denom
# Overlap Jaccard Similarity
def jaccard_overlap_pct(a: str, b: str) -> float:
    A = Counter(w.lower() for w in word_tokenize(a)) if a else Counter()
    B = Counter(w.lower() for w in word_tokenize(b)) if b else Counter()
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return 100.0 * inter / union if union else 0.0
#Containment (asymmetric “recall” on the anchor)
def containment_pct(a: str, b: str) -> float:
    A = Counter(w.lower() for w in word_tokenize(a)) if a else Counter()
    B = Counter(w.lower() for w in word_tokenize(b)) if b else Counter()
    if not A:
        return 0.0
    inter = len(A & B)
    return 100.0 * inter / len(A)
def classify_flag(pct: float, threshold: float = 40.0) -> str:
    return "high" if pct >= threshold else "low"

class SCPPDataset(Dataset):
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            dataset = json.load(f)   # data is a list of dicts
        tmp_rows=[]
        score_jaccard_list=[]
        score_containment_list=[]
        score_overlap_list=[]
        for record in dataset:
            anchor=record["caption"]
            paraphrase=record["caption2"]
            distractor=record["negative_caption"]
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

        # Compute "middle ground" thresholds (medians)
        # median = statistics.median(overlaps) if overlaps else 0.0
        # threshold=round(median, 2)

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


def robust_quantile_threshold(scores, target_high_frac=0.5):
    """
    Returns a threshold such that roughly target_high_frac of samples become 'high'.
    Works even when many values are identical (low variance).
    """
    n = len(scores)
    if n == 0:
        return 0.0
    scores_sorted = sorted(scores)
    # index where 'high' begins (top target_high_frac)
    # e.g., target_high_frac=0.5 -> split in half
    split_at = max(0, min(n - 1, int(round((1.0 - target_high_frac) * (n - 1)))))
    return scores_sorted[split_at]


def load_model(model_name="meta-llama/Meta-Llama-3-8B",access_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 if needed
        device_map="auto",
        token=access_token,  # Use access token if required
    )
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def free_all_torch():
    gc.collect()

    # Clear CUDA memory (if GPU available)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("✅ All PyTorch objects removed, CUDA cache cleared.")
if __name__ == "__main__":


    args = get_args()
    if(args.device=="auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    

    if(args.data_type=="auto"):
        file_list=["replace_att","replace_obj","swap_att","swap_obj","replace_rel"]
    else:
        file_list=[args.data_type]

    base_path=args.save_path+args.model_type+"/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for file in file_list:
        args.data_type=file  
        print("Running for ", file)
        args.save_path=base_path
        
        if("gemma" in args.model_type or "Llama" in args.model_type):
            if(str(args.mode)=="True"):
                args.save_path=args.save_path+"_"+file+"_averaging/"
            else:
                args.save_path=args.save_path+"_"+file+"_lasttoken/"
        else:
            args.save_path=args.save_path+"_"+file+"/"

        dataset= SCPPDataset(LOAD_PATH+ file+".json")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        if("Llama" in args.model_type and "Instruct" in args.model_type):
            print("note here")
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
        elif(args.model_type=="qwen"):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            qwen_counterfact_scpp(data_loader,args,device)
        elif(args.model_type=="e5"):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            e5_counterfact_scpp(data_loader,args,device)
        elif(args.model_type=="kalm"):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            kalm_counterfact_scpp(data_loader,args,device)
        elif(args.model_type=="promptriever"):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            promptretriever_counterfact_scpp(data_loader,args,device)
        # torch.cuda.empty_cache()
        free_all_torch()
        print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
        print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

    # for file in ["swap_att","swap_obj","replace_att","replace_obj","replace_rel"]:
    #     print("Running KaLM for ", file)
    #     kalm(device,sub_file=file)


    #     print("Running E5 for ", file)
    #     e5(device,sub_file=file)

    #     print("Running Qwen for ", file)
    #     qwen(device,sub_file=file)
    # print(default_cache_path)
    # hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    # print("HF_HOME =", os.getenv("HF_HOME"))
    # print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
    # print("HF_DATASETS_CACHE =", os.getenv("HF_DATASETS_CACHE"))
    # print("Default HF cache path (transformers):", default_cache_path)
    # print("Datasets cache (datasets):", hf_datasets_cache)