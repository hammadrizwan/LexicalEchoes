#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from quoraPD_analysis.distance_analysis import analyze_and_save_distances
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch
from sklearn.metrics import classification_report
from get_token  import *
"""
Llama 3.1 8B Instruct — Semantic Textual Similarity (STS), temperature 0
No argparse. No quantization. Chat-format prompting.

How to use:
  1) Ensure you've accepted model terms and are logged in to Hugging Face.
  2) Edit PAIRS below to your own data [(text_a, text_b), ...].
  3) Run: python llama31_sts_noargs.py

Notes:
  - Greedy decoding (do_sample=False, temperature=0.0) for determinism.
  - Uses chat template via tokenizer.apply_chat_template.
  - If you want bf16, set USE_BF16=True (hardware must support it).
"""
from torch.utils.data import Dataset, DataLoader
import json, ast
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
from scipy.stats import pearsonr, spearmanr
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
        # counter=5
        with open(tsv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                self.sentence1_list.append(_parse_field(r[s1_key]))
                self.sentence2_list.append(_parse_field(r[s2_key]))
                self.labels_list.append(int(r[label_key]))
                # counter-=1
                # if(counter==0):
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
    
# ------------------------- USER SETTINGS -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = get_token()
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"#update for 3.4. 17B
MAX_NEW_TOKENS = 96
# USE_BF16 = True   # set False to force fp16

# Replace with your own data: list of (text_a, text_b)
PAIRS: List[Tuple[str, str]] = [
    ("What are some unexpected things first-time visitors to Bangladesh notice ?", "What are some unexpected things , first-time visitors to Bangladesh notice ?"),
]

# ------------------------- PROMPTS -------------------------

SYSTEM_PROMPT = """
You are an expert judge of semantic textual equivalence (do NOT use entailment).
Given two texts A and B, output a single JSON object with exactly these fields:
  - "match": an integer 0 or 1, where 1 = A and B have the same meaning; 0 = otherwise.
Rules:
  • Judge strict equivalence of meaning. Only output 1 if A and B assert the same facts with the same scope and polarity.
  • Do NOT consider entailment: if one is more general/specific or adds/removes information (even if A ⇒ B or B ⇒ A), output 0.
  • Penalize any differences in entities, quantities, dates/times, locations, modality (may/must), negation, conditionals, or event order.
  • Ignore superficial wording (tense, punctuation, synonyms) that does not change truth conditions.
  • Topic relatedness without equivalence ⇒ 0. Contradictions ⇒ 0.
  • Be conservative: if uncertain, output 0.
  • Respond with JSON only. No explanations or extra keys.
Answer strictly in this template:
{
  "match": 0
}
or
{
  "match": 1
}

Examples:
A: Can you cancel a train ticket online after booking it ?
B: Can you book a train ticket online after canceling it ?
Output: {"match": 0}

A: Is there any difference between a psychiatrist and a psychologist ?
B: Is there any difference between a psychologist and a psychiatrist ?
Output: {"match": 1}

A: Why does water freeze faster in metal cups than in plastic cups ?
B: Why does water in plastic cups freeze faster than in metal cups ?
Output: {"match": 0}
"""

#add examples for non zero shot
#1,2,3,4,5 range and then threshold, 
#Binary version 1,0 vs true, false (check literature)
# Ternary version 1,0,-1 (check literature)
# define semantically equivalent for the model and provide examples 2


USER_PROMPT_TEMPLATE = (
    "Text A: \"\"\"{a}\"\"\"\n"
    "Text B: \"\"\"{b}\"\"\"\n\n"
    "Return JSON now"
)

# ------------------------- HELPERS -------------------------

def build_chat_messages(text_a: str, text_b: str):
    user_prompt = USER_PROMPT_TEMPLATE.format(a=text_a, b=text_b)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from a string."""
    s_stripped = s.strip()
    if s_stripped.startswith("{") and s_stripped.endswith("}"):
        try:
            return json.loads(s_stripped)
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# def select_dtype() -> torch.dtype:
#     if USE_BF16 and torch.cuda.is_available():
#         # If bf16 not supported, fall back to fp16
#         if torch.cuda.is_bf16_supported():  # type: ignore[attr-defined]
#             return torch.bfloat16
#     return torch.float16


# ------------------------- LOAD MODEL -------------------------



# ------------------------- INFERENCE -------------------------

@torch.inference_mode()
def score_pair(a: str, b: str, max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    """Return dict with raw_text, json (if parsed), and fields if present."""
    messages = build_chat_messages(a, b)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=False,            # greedy
        temperature=0.0,            # explicit, though ignored in greedy
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    gen_tokens = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    print("text",text)
    parsed = parse_json_from_text(text)
    print("parsed",parsed)
    result: Dict[str, Any] = {"raw_text": text, "json": parsed}
    if isinstance(parsed, dict):
        sim = parsed.get("match")
        # reason = parsed.get("reason")
        # if isinstance(sim, (int, float)):
        #     sim = max(0.0, min(5.0, float(sim)))
        result["similarity"] = sim

    return result
#Llama 4
@torch.inference_mode()
def judge_equivalence(A: str, B: str, processor, tok, model) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A: {A}\nB: {B}"},
    ]
    # Chat template → input IDs
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,     # deterministic
            do_sample=False
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    # print("text",text)
    # Robust JSON extraction (first {...} block)
    pattern = r'\{\s*"match"\s*:\s*([01])\s*\}'

    matches = re.findall(pattern, text)
    # print("matches:", matches)

    if matches:
        # take the last match, convert to int
        result = {"match": int(matches[-1])}
    else:
        result = {"match": None}

    return result

def process_pairs(pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    incorrect_pairs,correct_pairs = [],[]
    labels=[]
    predictions=[]
    pos_pairs=[]
    neg_pairs=[]
    with open("./quoraPD_analysis/llama4_generation_fewshot/distances.jsonl", "w", encoding="utf-8") as f:
        for row in tqdm(data_loader, desc="running"):
            # print(row)
            sentence1, sentence2, label = row
            # print(f"\n--- Pair {i} ---")
            # out = score_pair(sentence1, sentence2)
            out=judge_equivalence(sentence1, sentence2, processor, tokenizer, model)
            if(out["match"] is None):
                continue
            labels.append(label.item())
            predictions.append(out["match"])
            try:
                # print("label:",label.item(),"distance:",out["match"])
                f.write(json.dumps({"sentence1":sentence1,"sentence2":sentence2,"label":label.item(),"distance":out["match"]}) + "\n")
                # print("here",json.dumps({"sentence1":sentence1,"sentence2":sentence2,"label":label.item(),"distance":out["match"]}) + "\n")
                if(label==0):
                    incorrect_pairs.append(out["match"])
                    neg_pairs.append((sentence1,sentence2))
                else:
                    correct_pairs.append(out["match"]) 
                    pos_pairs.append((sentence1,sentence2))
            except Exception as e:
                print(out,e)
            # return None   
        # print("pos_pairs",pos_pairs,"neg_pairs",neg_pairs,"correct_pairs",correct_pairs,"incorrect_pairs",incorrect_pairs)
        report = classification_report(labels, predictions, digits=4)
        f.write("\n"+report)
        
        files = analyze_and_save_distances(
        incorrect_pairs,
        correct_pairs,
        title_prefix="PAWS",
        out_dir="./quoraPD_analysis/llama4_generation_fewshot/",
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
    return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # torch_dtype = select_dtype()

print(f"Loading tokenizer: {MODEL_ID}")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

print(f"Loading model: {MODEL_ID}")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     # torch_dtype=torch_dtype,
#     access_token=access_token,
#     device_map="auto",  # places weights on available GPUs; no quantization
# )
# model.eval()

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_auth_token=access_token
)
# model.eval()
for name, param in model.named_parameters():
    if param.device.type != "cuda":
        print("⚠️ Not on GPU:", name, param.device)
        break
else:
    print("✅ All model params on CUDA")
if __name__ == "__main__":
    data_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/QuoraPD/paws_out/train.tsv"
    dataset=PAWSDataset(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_text_pairs,  # keep strings as lists; labels as tensor
        # num_workers=0,
    )
    _ = process_pairs(data_loader)
