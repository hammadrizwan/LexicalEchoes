"""
Semantic Lens (SEM) — minimal, extensible codebase

Goal: From a model's intermediate hidden state h^(l)_t, learn a tiny affine map
per layer applied **after pooling** (pool-first) so the overall probe remains
strictly affine at the sentence level. We mean-pool with a mask to get a
sentence vector v^(l), then apply v' = A_l v + b_l, and finally compute a
scaled cosine for paraphrase vs. distractor. This measures how much semantic
information is *linearly available* at each layer with minimal inductive bias.

Data assumption (triplets): a dataloader yields batches with keys
  {
    "anchor":      List[str],
    "paraphrase":  List[str],   # positive
    "distractor":  List[str],   # negative
  }

Run (single GPU quick test; bring your own triplet loader):
python sem_lens.py \
  --model_id google/gemma-2-9b \
  --layers 1 3 6 12 \
  --epochs 1 --batch_size 4 --seq_len 256

Expected loader helper (you implement it):
from sem_dataset import build_sem_loaders
train_loader = build_sem_loaders(tokenizer, seq_len, batch_size, split="train")
# same for split="test"

Each batch must be a dict with three equal-length lists of strings: anchor/paraphrase/distractor.
"""
from __future__ import annotations
import math
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scpp_data import build_sem_loader_from_json
from counterfact_data import build_sem_loader_from_jsonl_counterfact
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

# ==========================
# Utilities
# ==========================

def exists(x):
    return x is not None


def init_linear_like_identity(dim: int) -> nn.Linear:
    """Linear(d->d) initialized at identity (helps stability/minimality)."""
    lin = nn.Linear(dim, dim, bias=True)
    with torch.no_grad():
        nn.init.eye_(lin.weight)
        lin.bias.zero_()
    return lin

def init_linear_like_identity_no_bias(dim: int) -> nn.Linear:
    lin = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        nn.init.eye_(lin.weight)
    return lin

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over sequence with attention mask.
    x: [B, T, d], mask: [B, T] (1 for tokens, 0 for pad)
    returns: [B, d]
    """
    m = mask.float().unsqueeze(-1)  # [B, T, 1]
    denom = m.sum(dim=1).clamp_min(1e-6)
    return (x * m).sum(dim=1) / denom


# ==========================
# Model Wrapper
# ==========================

class ModelWrapper:
    """
    Light wrapper over HF causal LMs to standardize:
      - Tokenization
      - Access to hidden states per layer (output_hidden_states=True)
    Works with Gemma and most GPT-like models.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,  # e.g., "flash_attention_2"
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,  # e.g., "auto"
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        kwargs = dict(
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        if exists(attn_implementation):
            kwargs["attn_implementation"] = attn_implementation
        if exists(device_map):
            kwargs["device_map"] = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
        self.config = config

        # Infer model dim
        self.d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if self.d_model is None:
            if "gemma" in model_id.lower():
                self.d_model = config.text_config.hidden_size
            else:
                raise ValueError("Could not infer hidden size; set manually.")

    def tokenize_batch(self, texts: List[str], add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

    @torch.no_grad()
    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs.hidden_states, outputs.logits


# ==========================
# SEM Lens (minimal linear translator)
# ==========================

@dataclass
class LensConfig:
    layers: List[int]
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 1
    l2_to_identity: float = 1e-3  # keep A close to I
    clip_grad_norm: Optional[float] = 1.0


class SemanticLens(nn.Module):
    """
    Pool-first linear probe.
      1) Masked mean of token states at layer l -> sentence vector v ∈ R^d
      2) Affine map v' = A_l v + b_l  (A_l ≈ I at init)
      3) L2-normalize sentence vectors and score with scaled cosine
    This keeps the transformation strictly at the sentence level.
    """

    def __init__(self, d_model: int, layers: List[int], cfg: LensConfig):
        super().__init__()
        self.d = d_model
        self.layers = layers
        self.cfg = cfg

        self.translators = nn.ModuleDict({str(l): init_linear_like_identity_no_bias(self.d) for l in layers})
        # self.biases = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(self.d)) for l in layers})# persistent=True if bias fixed to zero
        self.logit_scales = nn.ParameterDict({str(l): nn.Parameter(torch.tensor(2.0)) for l in layers})

    def _sentence_vec(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """h: [B,T,d], m: [B,T] -> [B,d]; strictly linear pooling."""
        return masked_mean(h, m)

    def _transform(self, v: torch.Tensor, layer: int) -> torch.Tensor:
        key = str(layer)
        return self.translators[key](v)# + self.biases[key]
    
    def sentence_repr(self, h: torch.Tensor, m: torch.Tensor, layer: int) -> torch.Tensor:
        """
        h: [B,T,d], m: [B,T] -> normalized sentence vector after affine map.
        """
        # masked mean pool (linear)
        v = masked_mean(h, m)                    # [B, d]
        v = self.translators[str(layer)](v) # TURN THIS ON AGAIN FOR TRANSFORM + getattr(self, "biases", {}).get(str(layer), 0.0)
        # L2-normalize -> unit vectors
        return F.normalize(v, dim=-1)
    
    def pair_score(self, h_a: torch.Tensor, h_b: torch.Tensor, m_a: torch.Tensor, m_b: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute scaled cosine between sentence-level affine transforms.
        returns: [B]
        """
        va = self._sentence_vec(h_a, m_a)  # [B,d]
        vb = self._sentence_vec(h_b, m_b)  # [B,d]
        va_p = self._transform(va, layer)
        vb_p = self._transform(vb, layer)
        # cosine needs normalization (nonlinear but applied post-affine)
        va_n = F.normalize(va_p, dim=-1)
        vb_n = F.normalize(vb_p, dim=-1)
        cos = (va_n * vb_n).sum(dim=-1)
        #computation below is ignored for eval
        s_min, s_max = 1.0, 20.0
        log_scale = torch.tanh(self.logit_scales[str(layer)])  # [-1,1]
        scale = s_min + (s_max - s_min) * (log_scale + 1) / 2
        return cos * scale

    def l2_identity_penalty(self) -> torch.Tensor:
        if self.cfg.l2_to_identity == 0:
            # choose a registered parameter to place the tensor on the correct device
            any_param = next(iter(self.translators.values())).weight
            return torch.tensor(0.0, device=any_param.device)
        total = 0.0
        for m in self.translators.values():
            total = total + (m.weight - torch.eye(m.weight.shape[0], device=m.weight.device)).pow(2).sum()
        return self.cfg.l2_to_identity * total


# ==========================
# Trainer
# ==========================

@dataclass
class TrainConfig:
    epochs: int = 1
    grad_accum: int = 1
    lr: float = 5e-4
    weight_decay: float = 1e-4
    clip_grad_norm: Optional[float] = 1.0
    log_every: int = 50
    num_workers: int = 2


class SEMTrainer:
    def __init__(self, model_wrap: ModelWrapper, lens: SemanticLens, train_cfg: TrainConfig, margin: float = 0.1):
        self.mw = model_wrap
        self.lens = lens
        self.cfg = train_cfg
        self.device = next(lens.parameters()).device
        self.margin = margin  # hinge margin in cosine units (0.05–0.2 typical)

        params = [p for p in lens.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def _tok(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.mw.tokenize_batch(texts)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # optional: build valid_mask to drop special tokens too
        if "special_tokens_mask" in enc:
            enc["valid_mask"] = enc["attention_mask"].bool() & (~enc["special_tokens_mask"].bool())
            enc["valid_mask"] = enc["valid_mask"].long()
        else:
            enc["valid_mask"] = enc["attention_mask"]
        return enc

    def step(self, batch: Dict[str, List[str]], layers: List[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.lens.train()

        anchors = batch["anchor"]
        paras   = batch["paraphrase"]
        negs    = batch["distractor"]

        enc_a = self._tok(anchors)
        enc_p = self._tok(paras)
        enc_n = self._tok(negs)

        # Backbone forward (frozen)
        with torch.no_grad():
            hs_a, _ = self.mw.forward_hidden_states(enc_a["input_ids"], enc_a.get("attention_mask"))
            hs_p, _ = self.mw.forward_hidden_states(enc_p["input_ids"], enc_p.get("attention_mask"))
            hs_n, _ = self.mw.forward_hidden_states(enc_n["input_ids"], enc_n.get("attention_mask"))

        losses = []
        logs: Dict[str, float] = {}

        for l in layers:
            # similarity scores (higher is better)
            s_pos = self.lens.pair_score(hs_a[l], hs_p[l], enc_a["attention_mask"], enc_p["attention_mask"], l)  # [B]
            s_neg = self.lens.pair_score(hs_a[l], hs_n[l], enc_a["attention_mask"], enc_n["attention_mask"], l)  # [B]

            # Hinge (margin ranking) loss on the gap: max(0, m - (s_pos - s_neg))
            gap = s_pos - s_neg
            loss_l = torch.clamp(self.margin - gap, min=0.0).mean()
            losses.append(loss_l)

            with torch.no_grad():
                acc = (gap > 0).float().mean().item()
                logs[f"acc_layer_{l}"] = float(acc)
                logs[f"gap_mean_layer_{l}"] = float(gap.mean().item())
                logs[f"loss_layer_{l}"] = float(loss_l.item())

        total_loss = torch.stack(losses).sum() + self.lens.l2_identity_penalty()
        total_loss.backward()
        return total_loss, logs

    def train_epochs(self, train_loader: DataLoader, layers: List[int], epochs: int):
        step_idx = 0
        for ep in range(epochs):
            for i, batch in enumerate(train_loader):
                self.opt.zero_grad(set_to_none=True)
                total_loss, logs = self.step(batch, layers)
                if exists(self.cfg.clip_grad_norm):
                    nn.utils.clip_grad_norm_(self.lens.parameters(), self.cfg.clip_grad_norm)
                self.opt.step()
                if hasattr(self.cfg, "log_every") and step_idx % getattr(self.cfg, "log_every", 50) == 0:
                    log_str = (f"ep {ep} step {step_idx} loss {float(total_loss):.4f} "
                               + " ".join([f"{k}:{v:.3f}" for k, v in logs.items()]))
                    print(log_str, flush=True)
                step_idx += 1

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, layers: List[int]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for batch in data_loader:
            enc_a = self._tok(batch["anchor"])
            enc_p = self._tok(batch["paraphrase"])
            enc_n = self._tok(batch["distractor"])

            hs_a, _ = self.mw.forward_hidden_states(enc_a["input_ids"], enc_a.get("attention_mask"))
            hs_p, _ = self.mw.forward_hidden_states(enc_p["input_ids"], enc_p.get("attention_mask"))
            hs_n, _ = self.mw.forward_hidden_states(enc_n["input_ids"], enc_n.get("attention_mask"))

            for l in layers:
                s_pos = self.lens.pair_score(hs_a[l], hs_p[l], enc_a["attention_mask"], enc_p["attention_mask"], l)
                s_neg = self.lens.pair_score(hs_a[l], hs_n[l], enc_a["attention_mask"], enc_n["attention_mask"], l)
                acc = (s_pos > s_neg).float().mean().item()
                key = f"ACC@layer_{l}"
                metrics[key] = metrics.get(key, 0.0) + acc
                counts[key] = counts.get(key, 0) + 1

        for k in list(metrics.keys()):
            metrics[k] /= max(1, counts[k])
        return metrics
    @torch.no_grad()
    def evaluate_norm_euclid_two_way(self, data_loader: DataLoader, layers: List[int]) -> Dict[str, float]:
        """
        Two-way consistency:
        (1) d(a,p) < d(a,n)
        (2) d(a,p) < d(p,n)
        Report per-layer ACC for each and their conjunction, plus mean distances.
        """
        metrics: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for batch in data_loader:
            enc_a = self._tok(batch["anchor"])
            enc_p = self._tok(batch["paraphrase"])
            enc_n = self._tok(batch["distractor"])

            hs_a, _ = self.mw.forward_hidden_states(enc_a["input_ids"], enc_a.get("attention_mask"))
            hs_p, _ = self.mw.forward_hidden_states(enc_p["input_ids"], enc_p.get("attention_mask"))
            hs_n, _ = self.mw.forward_hidden_states(enc_n["input_ids"], enc_n.get("attention_mask"))

            for l in layers:
                va = self.lens.sentence_repr(hs_a[l], enc_a["valid_mask"], l)
                vp = self.lens.sentence_repr(hs_p[l], enc_p["valid_mask"], l)
                vn = self.lens.sentence_repr(hs_n[l], enc_n["valid_mask"], l)

                d_ap = torch.linalg.norm(va - vp, dim=-1)
                d_an = torch.linalg.norm(va - vn, dim=-1)
                d_pn = torch.linalg.norm(vp - vn, dim=-1)

                acc_anchor = (d_ap < d_an).float().mean().item()
                acc_para   = (d_ap < d_pn).float().mean().item()
                acc_both   = ((d_ap < d_an) & (d_ap < d_pn)).float().mean().item()

                for key, val in {
                    f"ACC_anchor@layer_{l}": acc_anchor,
                    f"ACC_para@layer_{l}":   acc_para,
                    f"ACC_both@layer_{l}":   acc_both,
                    f"mean_d_ap@layer_{l}":  float(d_ap.mean().item()),
                    f"mean_d_an@layer_{l}":  float(d_an.mean().item()),
                    f"mean_d_pn@layer_{l}":  float(d_pn.mean().item()),
                }.items():
                    metrics[key]  = metrics.get(key, 0.0) + val
                    counts[key]   = counts.get(key, 0)   + 1

        # average over batches
        for k in list(metrics.keys()):
            metrics[k] /= max(1, counts[k])
        return metrics


# ==========================
# Dump utilities (optional)
# ==========================

@torch.no_grad()
def dump_sem_predictions(
    loader: DataLoader,
    mw: ModelWrapper,
    lens: SemanticLens,
    layers: List[int],
    limit: int,
    out_path: str,
):
    dev = next(lens.parameters()).device
    samples = []
    seen = 0

    for batch in loader:
        print(batch["anchor"],batch["paraphrase"],batch["distractor"])
        enc_a = mw.tokenize_batch(batch["anchor"])
        enc_p = mw.tokenize_batch(batch["paraphrase"])
        enc_n = mw.tokenize_batch(batch["distractor"])
        for enc in (enc_a, enc_p, enc_n):
            for k in enc: enc[k] = enc[k].to(dev)

        hs_a, _ = mw.forward_hidden_states(enc_a["input_ids"], enc_a.get("attention_mask"))
        hs_p, _ = mw.forward_hidden_states(enc_p["input_ids"], enc_p.get("attention_mask"))
        hs_n, _ = mw.forward_hidden_states(enc_n["input_ids"], enc_n.get("attention_mask"))

        B = enc_a["input_ids"].size(0)
        texts = list(zip(batch["anchor"], batch["paraphrase"], batch["distractor"]))

        for b in range(B):
            if seen >= limit:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model_id": mw.model_id,
                        "layers": layers,
                        "samples": samples,
                    }, f, ensure_ascii=False, indent=2)
                return

            rec = {
                "anchor": texts[b][0],
                "paraphrase": texts[b][1],
                "distractor": texts[b][2],
                "scores": {}
            }
            for l in layers:
                s_pos = lens.pair_score(hs_a[l][b:b+1], hs_p[l][b:b+1], enc_a["attention_mask"][b:b+1], enc_p["attention_mask"][b:b+1], l)
                s_neg = lens.pair_score(hs_a[l][b:b+1], hs_n[l][b:b+1], enc_a["attention_mask"][b:b+1], enc_n["attention_mask"][b:b+1], l)
                rec["scores"][str(l)] = {"pos": float(s_pos.item()), "neg": float(s_neg.item())}
            samples.append(rec)
            seen += 1

    # flush if not early-returned
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": mw.model_id,
            "layers": layers,
            "samples": samples,
        }, f, ensure_ascii=False, indent=2)




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

# ==========================
# Main
# ==========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/gemma-3-12b-pt")
    p.add_argument("--layers", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--l2_to_identity", type=float, default=1e-3)
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="./sem_lens_ckpt")
    p.add_argument("--dump_preds", type=int, default=100, help="If >0, dump N triplets with scores to JSON")
    args = p.parse_args()

    mw = ModelWrapper(
        model_id=args.model_id,
        device_map=args.device_map,
        attn_implementation=None,
    )

    
    build_sem_loader_from_jsonl_counterfact
    train_loader = build_sem_loader_from_jsonl_counterfact(
        jsonl_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/counterfact/Counterfact_OpenAI.jsonl",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        max_examples=3000
    )
    test_loader = build_sem_loader_from_jsonl_counterfact(
        jsonl_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/counterfact/Counterfact_OpenAI.jsonl",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        max_examples=None
    )
    # train_loader = build_sem_loader_from_json(
    #     json_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/scpp/data/swap_obj.json",
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=2,
    # )
    # test_loader = build_sem_loader_from_json(
    #     json_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/scpp/data/swap_obj.json",
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=2,
    # )

    lens_cfg = LensConfig(
        layers=args.layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        l2_to_identity=args.l2_to_identity,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lens = SemanticLens(d_model=mw.d_model, layers=args.layers, cfg=lens_cfg).to(device)

    trainer = SEMTrainer(
        model_wrap=mw,
        lens=lens,
        train_cfg=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        ),
    )

    print("Starting SEM training (paraphrase vs distractor)", flush=True)
    trainer.train_epochs(train_loader, layers=args.layers, epochs=args.epochs)

    print("Evaluating on test split", flush=True)
    metrics = trainer.evaluate_norm_euclid_two_way(test_loader, layers=args.layers)
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        "state_dict": lens.state_dict(),
        "config": asdict(lens_cfg),
        "model_id": args.model_id,
        "layers": args.layers,
        "d_model": mw.d_model,
    }, os.path.join(args.save_dir, "sem_lens.pt"))
    print(f"Saved SEM lens to {os.path.join(args.save_dir, 'sem_lens.pt')}")

    out_path = os.path.join(args.save_dir, "sem_triplet_scores.json")
    dump_sem_predictions(test_loader, mw, lens, args.layers, limit=args.dump_preds, out_path=out_path)
    print(f"Wrote sample triplet scores to {out_path}")


if __name__ == "__main__":
    main()
