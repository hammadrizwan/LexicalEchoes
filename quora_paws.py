from xml.parsers.expat import model

from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
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
import argparse
from model_files.qwen import qwen_quora_paws

from get_token import *
#----------------------------------------------------------------------------
# Section: File paths and argument parsing
#----------------------------------------------------------------------------
# region File paths and argument parsing
LAYER_MAPPING_DICT={"gemma-3-1b-pt":["model.layers.25"],"gemma-3-4b-pt":["model.language_model.layers.33"],"gemma-3-12b-pt":["model.language_model.layers.24"],
                    "gemma-3-1b-it":["model.layers.25"],"gemma-3-4b-it":["model.language_model.layers.33"],"gemma-3-12b-it":["model.language_model.layers.47"],
                    "Llama-3.2-1B":["model.layers.14"],"Llama-3.2-1B-Instruct":["model.layers.14"],"Llama-3.2-3B":["model.layers.27"],"Llama-3.2-3B-Instruct":["model.layers.27"],

                    }
ACCESS_TOKEN=get_token()


def get_args():
    parser = argparse.ArgumentParser(description="Lexical Bias Benchmarking PAWS")

    # Data and paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/QuoraPD/paws_out/train.tsv",
        help="Path to the dataset file or directory"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./quoraPD_analysis/",
        help="Path to save results (jsonl, csv, etc.)"
    )

    # Model settings
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen",
        choices=["qwen", "llama", "gemma-3-1b-pt","gemma-3-4b-pt","gemma-3-12b-pt","gemma-3-1b-it","gemma-3-4b-it","gemma-3-12b-it",
                 "Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"],
        help="Type of model to use for embeddings"
    )

    # Training / evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
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
    parser.add_argument(
        "--whitening_stats_path",
        type=str,
        default="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/common_pile_analysis/whitening_stats.pt",
        help="Path to whitening stats (if whitening is True)"
    )
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    return args
#endregion
#----------------------------------------------------------------------------
# Section: For reading the TSV files, adapted from PAWS-X code:
#----------------------------------------------------------------------------
# region Data Loading
def _parse_field(s: str) -> str:
    """
    Turn values like b'Will a message ... ?' into a normal str.
    If it's already a plain string, just return it.
    """
    s = s.strip()
    # print(ast.literal_eval(s).decode("utf-8"))
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (bytes, bytearray)):
            return lit.decode("utf-8", errors="replace")
        
        return str(lit)
    except Exception:
        # Fallback: remove a leading b and surrounding quotes if present
        if s.startswith(("b'", 'b"')):
            s = s[2:]
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
            s = s[1:-1]
        return s

class PAWSDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        s1_key: str = "sentence1",
        s2_key: str = "sentence2",
        label_key: str = "label",
    ):
        self.sentence1_list=[]
        self.sentence2_list=[]
        self.labels_list=[]
        counter=64*10
        with open(tsv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                self.sentence1_list.append(_parse_field(r[s1_key]))
                self.sentence2_list.append(_parse_field(r[s2_key]))
                self.labels_list.append(int(r[label_key]))
                # counter -= 1
                # if counter == 0:
                #     break

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        # returns (sentence1:str, sentence2:str, label:int)
        return self.sentence1_list[idx], self.sentence2_list[idx], self.labels_list[idx]

def collate_text_pairs(batch):
    # batch is a list of samples; each sample is (s1:str, s2:str, y:int)
    s1, s2, y = zip(*batch)  # tuples of length B
    if len(batch) == 1:
        # flatten for convenience when B=1
        return s1[0], s2[0], torch.tensor([y[0]], dtype=torch.long)
    else:
        return list(s1), list(s2), torch.tensor(y, dtype=torch.long)
#endregion 
#----------------------------------------------------------------------------
# Section: AutoModel loading and inference
#----------------------------------------------------------------------------
# region Autoloading 
def load_model(model_name="meta-llama/Meta-Llama-3-8B",access_token=None,devic="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 if needed
        device_map=device,
        token=access_token,  # Use access token if required
    )
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


#endregion
#----------------------------------------------------------------------------
# Section: Gemma model Instruction Tuned Analysis
#----------------------------------------------------------------------------
# region Gemma model Instruction Tuned Analysis
def build_batched_chat_inputs(tokenizer, texts, add_generation_prompt=True, device="auto"):
    """
    Build a batch of chat-formatted prompts for Gemma-IT using tokenizer.apply_chat_template.
    `texts` is a list[str] (already lowercased / formatted upstream if you want).
    Returns: dict with input_ids, attention_mask on device.
    """
    # Each item in the batch is a full conversation (list of messages)
    conversations = [
        [
            
            {"role": "user",
             "content": [{"type": "text", "text": "{}".format(t.lower())}]}
        ]
        for t in texts
    ]

    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        padding=True,                # important for batching
        return_dict=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in inputs.items()}

def gemma_it_quora_paws(data_loader,args,device):
    model_name = "google/"+args.model_type
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

    # prompt = "Explain the theory of relativity in simple terms."
    print("Loading model...")
    tokenizer, model = load_model(model_name,ACCESS_TOKEN,device)
    print(model)
    # return None
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    incorrect_pairs=[]
    correct_pairs=[]
    neg_pairs=[]
    pos_pairs=[]
    l=LAYER_MAPPING_DICT[args.model_type]#1b model
    # print(model)
    # l = ["model.language_model.layers.47"]#4b model
    # l=["model.language_model.layers.33"]#4b model
    model.eval()
    print(model)
 
    distance_file_path = os.path.join(args.save_path, "distances.jsonl")
    with open(distance_file_path, "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):
                    inputs=build_batched_chat_inputs(tokenizer, batch[0], add_generation_prompt=True, device=device)
                    outputs = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],output_hidden_states=True)
                    embeddings_sentence1=[ret[layer_fc1_vals].output[0][:, -1, :] for layer_fc1_vals in ret][0]

                    inputs=build_batched_chat_inputs(tokenizer, batch[1], add_generation_prompt=True, device=device)   
                    outputs = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],output_hidden_states=True)
                    embeddings_sentence2=[ret[layer_fc1_vals].output[0][:, -1, :] for layer_fc1_vals in ret][0]


                    embeddings_sentence1 = torch.nn.functional.normalize(embeddings_sentence1, p=2, dim=1)
                    embeddings_sentence2 = torch.nn.functional.normalize(embeddings_sentence2, p=2, dim=1)
                    distances = torch.norm(embeddings_sentence1 - embeddings_sentence2, p=2, dim=-1)  # [B]
                    distances = distances.detach().cpu().tolist()

                    # 2. Labels and text
                    labels = batch[2].detach().cpu().tolist()   # [B]
                    sents1 = batch[0]   # list of str
                    sents2 = batch[1]   # list of str

                    # 3. Append to correct/incorrect lists and write to file
                    for s1, s2, lab, dist in zip(sents1, sents2, labels, distances):
                        if lab == 0:
                            incorrect_pairs.append(dist)
                            neg_pairs.append((s1,s2))
                        else:
                            correct_pairs.append(dist)
                            pos_pairs.append((s1,s2))

                        f.write(json.dumps({
                            "sentence1": s1,
                            "sentence2": s2,
                            "label": int(lab),
                            "distance": float(dist)
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
#endregion
#----------------------------------------------------------------------------
# Section: Gemma model Pretrained Analysis
#----------------------------------------------------------------------------
# region Gemma model Pretrained Analysis
def build_batched_pt_inputs(tokenizer, texts, device="auto"):
    """
    Build a batch of prompts for Gemma-PT (base model) using plain tokenization.
    `texts` is a list[str]. If you were lowercasing upstream for IT, keep it for parity.
    Returns: dict with input_ids, attention_mask on device.
    """
    enc = tokenizer(
        [t.lower() for t in texts],   # keep/lift if you *don’t* want lowercase
        padding=True,                 # important for batching
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}
def gemma_pt_quora_paws(data_loader, args, device):
    """
    Same flow as your IT runner, but targets a base Gemma checkpoint (no chat template).
    Expects:
      - args.model_type like "gemma-2-2b" or "gemma-7b" (non -it)
      - load_model(), acess_token_gemma, LAYER_MAPPING_DICT, nethook, etc. available in scope
    """
    import os, json, torch
    from tqdm import tqdm

    model_name = "google/" + args.model_type
    
    # Load
    print("Loading model...")
    tokenizer, model = load_model(model_name, ACCESS_TOKEN, device)
    print(model)
    # return None
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()


    incorrect_pairs = []
    correct_pairs   = []
    neg_pairs       = []
    pos_pairs       = []

    # Choose layer(s) to trace
    l = LAYER_MAPPING_DICT[args.model_type]

    distance_file_path = os.path.join(args.save_path, "distances.jsonl")
    with open(distance_file_path, "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):
                    # --- sentence 1 ---
                    inputs = build_batched_pt_inputs(tokenizer, batch[0], device=device)
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    # grab the traced tensor(s); we mirror your IT logic (take last token)
                    embeddings_sentence1 = [ret[layer_key].output[0][:, -1, :] for layer_key in ret][0]

                    # --- sentence 2 ---
                    inputs = build_batched_pt_inputs(tokenizer, batch[1], device=device)
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    embeddings_sentence2 = [ret[layer_key].output[0][:, -1, :] for layer_key in ret][0]

                    # Normalize + L2 distances
                    embeddings_sentence1 = torch.nn.functional.normalize(embeddings_sentence1, p=2, dim=1)
                    embeddings_sentence2 = torch.nn.functional.normalize(embeddings_sentence2, p=2, dim=1)
                    distances = torch.norm(embeddings_sentence1 - embeddings_sentence2, p=2, dim=-1)  # [B]
                    distances = distances.detach().cpu().tolist()

                    # Labels/text
                    labels = batch[2].detach().cpu().tolist()  # [B]
                    sents1 = batch[0]  # list[str]
                    sents2 = batch[1]  # list[str]

                    # Collect + write
                    for s1, s2, lab, dist in zip(sents1, sents2, labels, distances):
                        if lab == 0:
                            incorrect_pairs.append(dist)
                            neg_pairs.append((s1, s2))
                        else:
                            correct_pairs.append(dist)
                            pos_pairs.append((s1, s2))

                        f.write(json.dumps({
                            "sentence1": s1,
                            "sentence2": s2,
                            "label": int(lab),
                            "distance": float(dist)
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
#endregion
#----------------------------------------------------------------------------
# Section: Llama model Instruct Analysis
#----------------------------------------------------------------------------
# region Llama model Instruct Analysis
def build_batched_llama_chat_inputs(tokenizer, texts, add_generation_prompt=True, device="auto"):
    """
    Build a batch of chat-formatted prompts for Llama-3.2-* Instruct using tokenizer.apply_chat_template.
    `texts` is a list[str].
    Returns: dict with input_ids, attention_mask on device.
    """
    conversations = [
        [
            {"role": "user", "content": [{"type": "text", "text": t.lower()}]}
        ]
        for t in texts
    ]
    enc = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def llama_it_quora_paws(data_loader, args, device):
    """
    Run PAWS-like pair distance extraction on Llama 3.2 Instruct checkpoints.
    Expects:
      - args.model_type in {"Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"} (without 'meta-llama/' prefix)
      - load_model(), acess_token_gemma, nethook, etc. defined in your env (mirrors your Gemma code)
      - args.save_path exists
    """
    import os, json, torch
    from tqdm import tqdm

    model_name = "meta-llama/" + args.model_type

    print("Loading model...")
    tokenizer, model = load_model(model_name, ACCESS_TOKEN, device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print(model)
    incorrect_pairs = []
    correct_pairs   = []
    neg_pairs       = []
    pos_pairs       = []
    # Pick a layer to trace.
    # If you already maintain a mapping, use it; otherwise pick mid layer.
    # return None
    l = LAYER_MAPPING_DICT[args.model_type]
    

    distance_file_path = os.path.join(args.save_path, "distances.jsonl")
    with open(distance_file_path, "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):
                    # Sentence 1
                    inputs = build_batched_llama_chat_inputs(tokenizer, batch[0], add_generation_prompt=True, device=device)
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    embeddings_sentence1=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

                    # Sentence 2
                    inputs = build_batched_llama_chat_inputs(tokenizer, batch[1], add_generation_prompt=True, device=device)
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    embeddings_sentence2=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

                    # L2 distance of unit-normalized last-token features
                    embeddings_sentence1 = torch.nn.functional.normalize(embeddings_sentence1, p=2, dim=1)
                    embeddings_sentence2 = torch.nn.functional.normalize(embeddings_sentence2, p=2, dim=1)
                    distances = torch.norm(embeddings_sentence1 - embeddings_sentence2, p=2, dim=-1).detach().cpu().tolist()


                    labels = batch[2].detach().cpu().tolist()  # [B]
                    sents1 = batch[0]  # list[str]
                    sents2 = batch[1]  # list[str]

                    # Collect + write
                    for s1, s2, lab, dist in zip(sents1, sents2, labels, distances):
                        if lab == 0:
                            incorrect_pairs.append(dist)
                            neg_pairs.append((s1,s2))
                        else:
                            correct_pairs.append(dist)
                            pos_pairs.append((s1,s2))

                        f.write(json.dumps({
                            "sentence1": s1,
                            "sentence2": s2,
                            "label": int(lab),
                            "distance": float(dist)
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
#endregion
#----------------------------------------------------------------------------
# Section: Llama model Pretrained Analysis
#----------------------------------------------------------------------------
# region Llama model Pretrained Analysis
def build_batched_llama_pt_inputs(tokenizer, texts, device="auto"):
    """
    Build a batch for Llama-3.2 base (non-instruct) using plain tokenization.
    """
    enc = tokenizer(
        [t.lower() for t in texts],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def llama_pt_quora_paws(data_loader, args, device):
    """
    Same as the instruct runner but targets base Llama 3.2 checkpoints (no chat template).
    Expects:
      - args.model_type in {"Llama-3.2-1B","Llama-3.2-3B"}
    """
    import os, json, torch
    from tqdm import tqdm

    model_name = "meta-llama/" + args.model_type

    print("Loading model...")
    tokenizer, model = load_model(model_name, ACCESS_TOKEN, device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print(model)

    l = LAYER_MAPPING_DICT[args.model_type]

    incorrect_pairs = []
    correct_pairs   = []
    neg_pairs       = []
    pos_pairs       = []

    distance_file_path = os.path.join(args.save_path, "distances.jsonl")
    with open(distance_file_path, "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):
                    # Sentence 1
                    inputs = build_batched_llama_pt_inputs(tokenizer, batch[0], device=device)
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    embeddings_sentence1=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

                    # Sentence 2
                    inputs = build_batched_llama_pt_inputs(tokenizer, batch[1], device=device)
                    mask = inputs["input_ids"] != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    embeddings_sentence2=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

                    # L2 distance of unit-normalized last-token features
                    embeddings_sentence1 = torch.nn.functional.normalize(embeddings_sentence1, p=2, dim=1)
                    embeddings_sentence2 = torch.nn.functional.normalize(embeddings_sentence2, p=2, dim=1)
                    distances = torch.norm(embeddings_sentence1 - embeddings_sentence2, p=2, dim=-1).detach().cpu().tolist()


                    labels = batch[2].detach().cpu().tolist()  # [B]
                    sents1 = batch[0]  # list[str]
                    sents2 = batch[1]  # list[str]

                    # Collect + write
                    for s1, s2, lab, dist in zip(sents1, sents2, labels, distances):
                        if lab == 0:
                            incorrect_pairs.append(dist)
                            neg_pairs.append((s1,s2))
                        else:
                            correct_pairs.append(dist)
                            pos_pairs.append((s1,s2))

                        f.write(json.dumps({
                            "sentence1": s1,
                            "sentence2": s2,
                            "label": int(lab),
                            "distance": float(dist)
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

# endregion

#----------------------------------------------------------------------------
# Section: Llama old models
#----------------------------------------------------------------------------
# region Llama old models
def llama_quora_paws(data_loader,args,device):
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


    
    # prompt = "Explain the theory of relativity in simple terms."
    print("Loading model...")
    tokenizer, model = load_model(model_name,ACCESS_TOKEN,device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    incorrect_pairs=[]
    correct_pairs=[]
    neg_pairs=[]
    pos_pairs=[]
    l=["model.layers.31"]
    model.eval()
    distance_file_path = os.path.join(args.save_path, "distances.jsonl")
    with open(distance_file_path, "w", encoding="utf-8") as f:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="running"):
                    # print(batch[0])
                    sentences1 = [string.lower() for string in batch[0]]
                    inputs=tokenizer(sentences1, return_tensors="pt",padding=True)["input_ids"].to(device)    
                    mask = inputs != tokenizer.pad_token_id
                    last_index = mask.sum(dim=1) - 1
                    outputs = model(inputs, output_hidden_states=True)
                    embeddings_sentence1=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]

                    sentences2 = [string.lower() for string in batch[1]]
                    inputs=tokenizer(sentences2, return_tensors="pt",padding=True)["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    mask = inputs != tokenizer.pad_token_id#create mask for non padding tokens
                    last_index = mask.sum(dim=1) - 1 #get last non padding token index for each sentence
                    embeddings_sentence2=[ret[layer_fc1_vals].output[0][torch.arange(ret[layer_fc1_vals].output[0].size(0)), last_index,:] for layer_fc1_vals in ret][0]


                    embeddings_sentence1 = torch.nn.functional.normalize(embeddings_sentence1, p=2, dim=1)
                    embeddings_sentence2 = torch.nn.functional.normalize(embeddings_sentence2, p=2, dim=1)
                    distances = torch.norm(embeddings_sentence1 - embeddings_sentence2, p=2, dim=-1)  # [B]
                    distances = distances.detach().cpu().tolist()

                    # 2. Labels and text
                    labels = batch[2].detach().cpu().tolist()   # [B]
                    sents1 = batch[0]   # list of str
                    sents2 = batch[1]   # list of str

                    # 3. Append to correct/incorrect lists and write to file
                    for s1, s2, lab, dist in zip(sents1, sents2, labels, distances):
                        if lab == 0:
                            incorrect_pairs.append(dist)
                            neg_pairs.append((s1,s2))
                        else:
                            correct_pairs.append(dist)
                            pos_pairs.append((s1,s2))

                        f.write(json.dumps({
                            "sentence1": s1,
                            "sentence2": s2,
                            "label": int(lab),
                            "distance": float(dist)
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
#endregion  
#----------------------------------------------------------------------------
# Section: Qwen 8b Embedding Model Analysis
#----------------------------------------------------------------------------
#region Qwen 8b Embedding Model Analysis

# endregion
#----------------------------------------------------------------------------
                
if __name__ == "__main__":
    args = get_args()
    if(args.device=="auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    args.save_path=args.save_path+args.model_type+"/"
    data_path=args.dataset_path
    dataset=PAWSDataset(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_text_pairs,  # keep strings as lists; labels as tensor
        # num_workers=0,
    )
    if("Llama" in args.model_type and "Instruct" in args.model_type):
        print("note here")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        llama_it_quora_paws(data_loader,args,device)
    elif("Llama" in args.model_type and "Instruct" not in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        llama_pt_quora_paws(data_loader,args,device)
    elif("gemma" in args.model_type and "pt" in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        gemma_pt_quora_paws(data_loader,args,device)
    elif("gemma" in args.model_type and "it" in args.model_type):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        gemma_it_quora_paws(data_loader,args,device)
    elif(args.model_type=="qwen"):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        qwen_quora_paws(data_loader,args,device)
    