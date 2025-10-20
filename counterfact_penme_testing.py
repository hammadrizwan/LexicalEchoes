from xml.parsers.expat import model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from transformers.utils import default_cache_path
import nethook
import os,json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import linecache
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from LexicalBias.Lexical_Semantic_Quantification.model_files.qwen import qwen_test_direct_counterfact,qwen_test_stacked_counterfact
from LexicalBias.Lexical_Semantic_Quantification.model_files.llama_functions import llama_embeddings_analysis_counterfact_average,llama_embeddings_analysis_counterfact_lasttoken
from LexicalBias.Lexical_Semantic_Quantification.model_files.gemma_functions import gemma_embeddings_analysis_counterfact_average,gemma_embeddings_analysis_counterfact_lasttoken
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


class CounterFactDataset(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as jsonl_file:
          lines = jsonl_file.readlines()
        self.data = [json.loads(line) for line in lines]
        # self.tokenizer=tokenizer
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

def cosine_distance(a, b):
    return 1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def gemma():
    model_name = "google/gemma-7b-it"
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
    print("Model loaded.",model)
    # print("Generating response...")
    # response = generate_response(prompt, tokenizer, model)
    # print("Run Embeddigs Analysis Counterfact...")
    file_path_counterfact="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/Counterfact_OpenAI.jsonl"
    # file_save_path="./counterfact_gemma_lexical_bias_violations_lasttoken.jsonl"
    # gemma_embeddings_analysis_counterfact_lasttoken(file_path=file_path_counterfact,model=model,tokenizer=tokenizer,file_save_path=file_save_path,device="cuda:0")
    file_save_path="./counterfact_gemma_lexical_bias_violations_average.jsonl"
    gemma_embeddings_analysis_counterfact_average(file_path=file_path_counterfact,model=model,tokenizer=tokenizer,file_save_path=file_save_path,device="cuda:0")  
    # print("\n=== Model Output ===")
    # print(response)
def llama():
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
    # print("Model loaded.",model)
    # print("Generating response...")
    # response = generate_response(prompt, tokenizer, model)
    # print("Run Embeddigs Analysis Counterfact...")
    file_path_counterfact="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/Counterfact_OpenAI.jsonl"
    file_save_path="./counterfact_llama_lexical_bias_violations_lasttoken.jsonl"
    llama_embeddings_analysis_counterfact_lasttoken(file_path=file_path_counterfact,model=model,tokenizer=tokenizer,file_save_path=file_save_path,device="cuda:0")
    file_save_path="./counterfact_llama_lexical_bias_violations_average.jsonl"
    llama_embeddings_analysis_counterfact_average(file_path=file_path_counterfact,model=model,tokenizer=tokenizer,file_save_path=file_save_path,device="cuda:0")  
    # print("\n=== Model Output ===")
    # print(response)

def qwen(device):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B").to(device)
    # queries = [
    # "Angola is located in",
    # ]
    # documents = [
    #     "Mozambique is in",
    #     "Angola is a part of the continent of",
    # ]
    file_path_counterfact="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/Counterfact_OpenAI.jsonl"
    file_save_path="./counterfact_analysis/counterfact_qwen_lexical_bias_direct.jsonl"
    qwen_test_direct_counterfact(file_path=file_path_counterfact,model=model,file_save_path=file_save_path,device="cuda:0")
    # query_embeddings = model.encode(queries)
    # document_embeddings = model.encode(documents)
    # print("Query Embeddings:", query_embeddings)
    # print("Query Embeddings:", query_embeddings[0].shape)
    # Compute the (cosine) similarity between the query and document embeddings
    # similarity = model.similarity(query_embeddings, document_embeddings)
    # print(similarity)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen(device)
    # print(default_cache_path)
    # hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    # print("HF_HOME =", os.getenv("HF_HOME"))
    # print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
    # print("HF_DATASETS_CACHE =", os.getenv("HF_DATASETS_CACHE"))
    # print("Default HF cache path (transformers):", default_cache_path)
    # print("Datasets cache (datasets):", hf_datasets_cache)