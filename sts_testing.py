from xml.parsers.expat import model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import ast
from typing import Tuple, List
# from transformers.utils import default_cache_path
import nethook
import os,json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import linecache
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from model_files.qwen import qwen_get_embeddings
from quoraPD_analysis.distance_analysis import analyze_and_save_distances
# from LexicalBias.Lexical_Semantic_Quantification.model_files.llama_functions import llama_embeddings_analysis_counterfact_average,llama_embeddings_analysis_counterfact_lasttoken
# from LexicalBias.Lexical_Semantic_Quantification.model_files.gemma_functions import gemma_embeddings_analysis_counterfact_average,gemma_embeddings_analysis_counterfact_lasttoken
from scipy.stats import pearsonr, spearmanr

class STS(Dataset):
    def __init__(
        self,
        tsv_path: str,
    ):
        self.sentence1_list=[]
        self.sentence2_list=[]
        self.scores=[]
        with open(tsv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                # print(line)
                self.sentence1_list.append(line["sentence1"])
                self.sentence2_list.append(line["sentence2"])
                self.scores.append(float(line["score"]))
                

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # returns (sentence1:str, sentence2:str, label:int)
        return self.sentence1_list[idx], self.sentence2_list[idx], self.scores[idx]

def collate_text_pairs(batch):
    # batch is a list of samples; each sample is (s1:str, s2:str, y:int)
    s1, s2, y = zip(*batch)  # tuples of length B
    if len(batch) == 1:
        # flatten for convenience when B=1
        return s1[0], s2[0], torch.tensor([y[0]], dtype=torch.long)
    else:
        return list(s1), list(s2), torch.tensor(y, dtype=torch.long)
      
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

def gemma(data_loader,devices):
    model_name = "google/gemma-3-1b-it"
    # GemmaForCausalLM
    #     └── model (GemmaModel)
    #         └── embed_tokens
    #         └── layers (ModuleList)
    #             └── [i] (GemmaDecoderLayer)
    #                 └── self_attn (GemmaAttention)
    #                     └── q_proj / k_proj / v_proj / o_proj
    #                 └── mlp (GemmaMLP)
    #                     └── gate_proj / up_proj / down_proj / act_fn
    #                 └── input_layernorm
    #                 └── post_attention_layernorm
    #         └── norm
    #         └── rotary_emb
    #     └── lm_head
    acess_token_gemma= "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"
    # prompt = "Explain the theory of relativity in simple terms."
    print("Loading model...")
    tokenizer, model = load_model(model_name,acess_token_gemma)
    incorrect_pairs=[]
    correct_pairs=[]
    l=["model.layers.25"]
    model.eval()
    print(model)
    with open("./quoraPD_analysis/gemma/distances.jsonl", "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):

                    inputs=tokenizer(batch[0], return_tensors="pt").to(device)    
                    outputs = model(**inputs, output_hidden_states=True)
                    embedding_sentence1=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]
                
                    inputs=tokenizer(batch[1], return_tensors="pt").to(device)    
                    outputs = model(**inputs, output_hidden_states=True)
                    embedding_sentence2=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]

                    embedding_sentence1 = torch.nn.functional.normalize(embedding_sentence1, p=2, dim=0)
                    embedding_sentence2 = torch.nn.functional.normalize(embedding_sentence2, p=2, dim=0)
                    
                    distance  = torch.norm(embedding_sentence1 - embedding_sentence2, p=2).item()
                    if(batch[2].item()==0):
                        incorrect_pairs.append(distance)
                    else:
                        correct_pairs.append(distance)  
                    f.write(json.dumps({"sentence1":batch[0],"sentence2":batch[1],"label":batch[2].item(),"distance":distance}) + "\n")
    files = analyze_and_save_distances(
            incorrect_pairs,
            correct_pairs,
            title_prefix="Qwen pairwise distances",
            out_dir="./quoraPD_analysis/gemma/"   # change if you want a different folder
        )
def llama(data_loader,device):
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # LlamaForCausalLM
        # └── model (LlamaModel)
        #     ├── embed_tokens (Embedding)
        #     ├── layers (ModuleList)
        #     │   └── [i] (LlamaDecoderLayer)
        #     │       ├── self_attn (LlamaAttention)
        #     │       │   ├── q_proj (Linear)
        #     │       │   ├── k_proj (Linear)
        #     │       │   ├── v_proj (Linear)
        #     │       │   └── o_proj (Linear)
        #     │       ├── mlp (LlamaMLP)
        #     │       │   ├── gate_proj (Linear)
        #     │       │   ├── up_proj (Linear)
        #     │       │   ├── down_proj (Linear)
        #     │       │   └── act_fn (SiLU)
        #     │       ├── input_layernorm (LlamaRMSNorm)
        #     │       └── post_attention_layernorm (LlamaRMSNorm)
        #     ├── norm (LlamaRMSNorm)
        #     └── rotary_emb (LlamaRotaryEmbedding)
        # └── lm_head (Linear)

    acess_token_gemma= "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"
    # prompt = "Explain the theory of relativity in simple terms."
    print("Loading model...")
    tokenizer, model = load_model(model_name,acess_token_gemma)
    incorrect_pairs=[]
    correct_pairs=[]
    l=["model.layers.31"]
    model.eval()
    with open("./quoraPD_analysis/llama/distances.jsonl", "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):

                    inputs=tokenizer(batch[0].lower(), return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    embedding_sentence1=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]
                
                    inputs=tokenizer(batch[1].lower(), return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    embedding_sentence2=[ret[layer_fc1_vals].output[0][:, -1, :].view(-1) for layer_fc1_vals in ret][0]


                    embedding_sentence1 = torch.nn.functional.normalize(embedding_sentence1, p=2, dim=0)#dim 0 fine when single input
                    embedding_sentence2 = torch.nn.functional.normalize(embedding_sentence2, p=2, dim=0)
                    distance  = torch.norm(embedding_sentence1 - embedding_sentence2, p=2).item()
                    if(batch[2].item()==0):
                        incorrect_pairs.append(distance)
                    else:
                        correct_pairs.append(distance)  
                    f.write(json.dumps({"sentence1":batch[0],"sentence2":batch[1],"label":batch[2].item(),"distance":distance}) + "\n")
    files = analyze_and_save_distances(
            incorrect_pairs,
            correct_pairs,
            title_prefix="Qwen pairwise distances",
            out_dir="./quoraPD_analysis/llama/"   # change if you want a different folder
        )

def qwen(data_loader,device):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B").to(device)
    model.eval()
    labels=[]
    prediction_distances=[]
    prediction_similarities=[]
    with open("./sts/results_qwen.jsonl", "w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="running"):
                    # print(batch)
                    embedding_sentence1=qwen_get_embeddings(model,batch[0])
                    # print(embedding_sentence1.shape)
                    embedding_sentence2=qwen_get_embeddings(model,batch[1])

                    embedding_sentence1 = torch.nn.functional.normalize(embedding_sentence1, p=2, dim=1)
                    embedding_sentence2 = torch.nn.functional.normalize(embedding_sentence2, p=2, dim=1)

                    similarities,similarities_scaled,distances,distances_scaled=pairwise_metrics(embedding_sentence1, embedding_sentence2)
                    # print(distances.item(),similarities.item())
                    for s1,s2,score,dist,sim in zip(batch[0],batch[1],batch[2],distances_scaled.cpu().numpy().tolist(),similarities_scaled.cpu().numpy().tolist()):
                        f.write(json.dumps({"sentence1":s1,"sentence2":s2,"label":score.cpu().numpy().tolist(),"distance":dist/5,"cosine sim":sim/5}) + "\n")
                    labels.extend(batch[2].cpu().numpy().tolist())
                    prediction_distances.extend(distances_scaled.cpu().numpy().tolist())
                    prediction_similarities.extend(similarities_scaled.cpu().numpy().tolist())
        pearson_corr_d, pearson_p_d = pearsonr(labels, prediction_distances)
        pearson_corr_s, pearson_p_s = pearsonr(labels, prediction_similarities)

        # Spearman correlation
        spearman_corr_d, spearman_p_d = spearmanr(labels, prediction_distances)
        spearman_corr_s, spearman_p_s = spearmanr(labels, prediction_similarities)
        f.write(json.dumps({"pearson_corr_d":pearson_corr_d,"pearson_p_d":pearson_p_d,"pearson_corr_s":pearson_corr_s,"pearson_p_s":pearson_p_s,"spearman_corr_d":spearman_corr_d,"spearman_p_d":spearman_p_d,"spearman_corr_s":spearman_corr_s,"spearman_p_s":spearman_p_s}) + "\n")



def pairwise_metrics(E1: torch.Tensor, E2: torch.Tensor, eps: float = 1e-8):
    assert E1.shape == E2.shape and E1.dim() == 2, "E1 and E2 must be [N, D] and same shape"

    # Cosine similarity per row ([-1, 1])
    cos_sim = F.cosine_similarity(E1, E2, dim=1, eps=eps)  # shape [N]

    # Rescale cosine sim to [0, 5]
    cos_scaled = 2.5 * (cos_sim + 1.0)

    # Euclidean distance per row ([0, 2] if E1, E2 are normalized)
    euclid = torch.linalg.norm(E1 - E2, dim=1)             # shape [N]

    # Rescale Euclidean distance to [0, 5] similarity
    euclid_scaled = 5.0 * (1.0 - euclid / 2.0)

    return cos_sim, cos_scaled, euclid, euclid_scaled
            
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/sts/stsb_train.tsv"
    dataset=STS(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        # collate_fn=collate_text_pairs,  # keep strings as lists; labels as tensor
        # num_workers=0,
    )
    qwen(data_loader,device)
    # for row in data_loader:
    #     print(row)
    #     print(row[2].item()==0)
    #     break
    # qwen(data_loader,device)
    # llama(data_loader,device)
    # gemma(data_loader,device)

    # print(default_cache_path)
    # hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    # print("HF_HOME =", os.getenv("HF_HOME"))
    # print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
    # print("HF_DATASETS_CACHE =", os.getenv("HF_DATASETS_CACHE"))
    # print("Default HF cache path (transformers):", default_cache_path)
    # print("Datasets cache (datasets):", hf_datasets_cache)