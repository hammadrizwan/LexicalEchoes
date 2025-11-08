"""
Reverse Lexical Lens — minimal, extensible codebase

Goal: From a model's intermediate hidden state h^(l)_t, learn an affine map that
scores the *input token* via the model's input embedding matrix E.


Run (single GPU quick test):
python reverse_lexical_lens.py \
  --model_id google/gemma-2-9b \
  --layers 0 1 3 6 12 \
  --dataset_path path/to/text.txt \
  --epochs 1 --batch_size 4 --seq_len 512

"""
from __future__ import annotations
import torch.nn.functional as F
from wiki_texts import build_wikitext_loaders
import math
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple
import sys
sys.path.append("/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/")
from get_token import get_token   
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    """Linear(d->d) initialized close to identity (useful for stability)."""
    lin = nn.Linear(dim, dim, bias=True)
    with torch.no_grad():
        nn.init.eye_(lin.weight)
        lin.bias.zero_()
    return lin


# ==========================
# Model Wrapper
# ==========================

class ModelWrapper:
    """
    A light wrapper over HF causal LMs that standardizes:
      - Tokenization & batching
      - Access to the input embedding matrix E
      - Extraction of layer hidden states at consistent hook points

    Works with Gemma and most GPT-like models as long as the model returns
    hidden_states when output_hidden_states=True.
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
            # For causal LMs, set pad to eos for batching convenience
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map is None:  # not using accelerate-style placement
            self.model.to(self.device)

        self.model.eval()
        self.config = config

        # Infer model dim and vocab size
        self.d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        print(config)
        if self.d_model is None:
            if "gemma" in model_id.lower():
                self.d_model = config.text_config.hidden_size
            else:
                raise ValueError("Could not infer hidden size from config; please set manually.")
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @property
    def E_weight(self) -> torch.Tensor:
        # Shape: [V, d]
        return self.input_embeddings.weight

    def tokenize_batch(self, texts: List[str], seq_len: int, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return enc

    @torch.no_grad()
    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Returns (hidden_states, last_logits)
        - hidden_states: tuple of layer outputs including embedding layer at index 0
          Shape per item: [B, T, d]
        - last_logits: [B, T, V]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states  # type: ignore
        logits = outputs.logits  # [B, T, V]
        return hidden_states, logits




# ==========================
# Reverse Lexical Lens
# ==========================

@dataclass
class LensConfig:
    layers: List[int]
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1
    temperature_init: float = 1.0
    l2_to_identity: float = 0.0  # regularize A toward I
    clip_grad_norm: Optional[float] = 1.0
    use_amp: bool = True


class ReverseLexicalLens(nn.Module):
    """
    For each selected layer l, learns an affine translator (A_l, b_l):
       h' = A_l h + b_l  (A_l in R^{dxd}, b_l in R^d)
    Predict logits via the model's input embedding matrix E: logits = E h'
    We optionally apply a temperature per layer.
    """

    def __init__(self, d_model: int, vocab_size: int, E: torch.Tensor, layers: List[int], cfg: LensConfig):
        super().__init__()
        self.d = d_model
        self.vocab_size = vocab_size
        self.layers = layers
        self.cfg = cfg

        # Register E as a buffer (frozen). Shape [V, d]
        self.register_buffer("E", E.detach().clone(), persistent=False)# un normalized one has dependence on norm of E

        # Normalized E for cosine classifier
        E_norm = F.normalize(self.E, dim=1)  # [V, d] row-wise normalize
        self.register_buffer("E_norm", E_norm, persistent=False)

        # # Per-layer logit scales (start around exp(2)=~7.39, mild sharpness)
        self.logit_scales = nn.ParameterDict({
            str(l): nn.Parameter(torch.tensor(2.0))  # this is log_scale before bounding
            for l in layers
        })
        self.translators = nn.ModuleDict()
        self.temperatures = nn.ParameterDict()

        for l in layers:
            mod = init_linear_like_identity(self.d)
            self.translators[str(l)] = mod

            t = nn.Parameter(torch.tensor(float(cfg.temperature_init)))
            self.temperatures[str(l)] = t

        # Simple bias per layer
        self.biases = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(self.d)) for l in layers})

    def forward_layer(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """
        h: [B, T, d] 
        logits: [B, T, V] for a single layer.
        Perform the affine translation and return logits.
        Normalization and scaling to improve training and remove sensitivity to norm of Embedding and hidden states.
        """
        key = str(layer)
        A = self.translators[key]#extract layer A
        b = self.biases[key]#extract layer bias
        

        h2 = A(h) + b
        # Normalize features
        h2 = F.normalize(h2, dim=-1)  # [B, T, d]

        # Bounded scale: logit_scale_raw -> logit_scale in [s_min, s_max] GPT based other wise there is temp
        # (tanh keeps it stable; adjust bounds if desired)
        # #Either use fixed temperature or learned temperature
        s_min, s_max = 1.0, 20.0#
        log_scale = torch.tanh(self.logit_scales[key])  # [-1, 1]
        scale = s_min + (s_max - s_min) * (log_scale + 1) / 2  # [s_min, s_max]
        temp = torch.clamp(self.temperatures[key], min=1e-3)

        # [B, T, d] x [d, V] -> [B, T, V]
        logits = torch.matmul(h2, self.E_norm.t())  *scale #/ temp
        return logits

    def l2_identity_penalty(self) -> torch.Tensor:
        if self.cfg.l2_to_identity == 0:
            return torch.tensor(0.0, device=self.E.device)
        total = 0.0
        for m in self.translators.values():  
            total = total + (m.weight - torch.eye(m.weight.shape[0], device=m.weight.device)).pow(2).sum()
           
        return self.cfg.l2_to_identity * total



# ==========================
# Trainer
# ==========================

@dataclass
class TrainConfig:
    seq_len: int = 512
    epochs: int = 1
    grad_accum: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: Optional[float] = 1.0
    use_amp: bool = True
    log_every: int = 50
    num_workers: int = 2


class ReverseLexicalLensTrainer:
    def __init__(self, model_wrap: ModelWrapper, lens: ReverseLexicalLens, train_cfg: TrainConfig):
        self.mw = model_wrap
        self.lens = lens
        self.cfg = train_cfg
        self.device = next(lens.parameters()).device

        params = [p for p in lens.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.mw.tokenizer.pad_token_id)

    def step(self, batch: Dict[str, torch.Tensor], layers: List[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.lens.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Backbone forward without grads (we only train the small lens)
        with torch.no_grad():
            hidden_states, _ = self.mw.forward_hidden_states(
                input_ids=input_ids, attention_mask=attention_mask
            )  # tuple(Tensors[B,T,d])

        labels = batch["labels"].to(self.device)

        # Safety checks
        assert labels.dtype in (torch.long, torch.int64), "CrossEntropyLoss needs integer class indices"
        # (E is a buffer; lens params require grad by construction)

        losses = []
        logs: Dict[str, float] = {}

        for l in layers:
            h = hidden_states[l]                      # [B, T, d] (no grad needed for h)
            logits = self.lens.forward_layer(h, l)    # [B, T, V], depends on lens params -> requires grad
            loss_l = self.ce(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses.append(loss_l)
            logs[f"ce_layer_{l}"] = float(loss_l.detach())

        total_loss = torch.stack(losses).sum()

        # Optional regularizer toward identity
        reg = self.lens.l2_identity_penalty()
        if reg is not None:
            total_loss = total_loss + reg

        # Standard backward (no AMP, no scaler)
        total_loss.backward()

        return total_loss, logs

    def train_epochs(self, train_loader: DataLoader, layers: List[int], epochs: int):
        step_idx = 0
        for ep in range(epochs):
            for i, batch in enumerate(train_loader):
                # zero grads BEFORE backward for clarity (either order works if you use set_to_none)
                self.opt.zero_grad(set_to_none=True)

                total_loss, logs = self.step(batch, layers)

                if exists(self.cfg.clip_grad_norm):
                    nn.utils.clip_grad_norm_(self.lens.parameters(), self.cfg.clip_grad_norm)

                self.opt.step()

                if step_idx % self.cfg.log_every == 0:
                    log_str = (
                        f"ep {ep} step {step_idx} loss {float(total_loss):.4f} "
                        + " ".join([f"{k}:{v:.3f}" for k, v in logs.items()])
                    )
                    print(log_str, flush=True)
                step_idx += 1

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, layers: List[int], topk: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
        """
        Returns per-layer CE and Top-k ACC, ignoring PAD positions.
        Keys look like CE@layer_3, Top1@layer_3, etc.
        """
        pad_id = self.mw.tokenizer.pad_token_id
        metrics: Dict[str, float] = {}

        for batch in data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            hidden_states, _ = self.mw.forward_hidden_states(input_ids=input_ids, attention_mask=attention_mask)

            # Mask for valid positions (non-PAD)
            valid = (labels != pad_id)

            for l in layers:
                h = hidden_states[l]
                logits = self.lens.forward_layer(h, l)

                # CE (ignore PAD internally)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                    ignore_index=pad_id, reduction='mean'
                )
                key_ce = f"CE@layer_{l}"
                metrics[key_ce] = metrics.get(key_ce, 0.0) + float(loss)
                metrics[f"_counts_{key_ce}"] = metrics.get(f"_counts_{key_ce}", 0) + 1

                # Top-k accuracy per layer, ignoring PAD positions
                for k in topk:
                    topk_idx = logits.topk(k, dim=-1).indices  # [B, T, k]
                    match = (topk_idx == labels.unsqueeze(-1)).any(dim=-1) & valid
                    correct = match.sum().item()
                    total_valid = valid.sum().item()
                    key = f"Top{k}@layer_{l}"
                    metrics[key] = metrics.get(key, 0.0) + (correct / max(1, total_valid))
                    metrics[f"_counts_{key}"] = metrics.get(f"_counts_{key}", 0) + 1

        # Average over batches
        final: Dict[str, float] = {}
        for k, v in metrics.items():
            if k.startswith("_counts_"):
                continue
            count = metrics.get(f"_counts_{k}", 1)
            final[k] = v / count
        return final





def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/gemma-3-12b-pt")
    # p.add_argument("--dataset_path", type=str, required=True, help="Plain text file to stream tokens from")
    p.add_argument("--layers", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47], help="Indices into hidden_states tuple")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--l2_to_identity", type=float, default=0.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="./revlexlens_ckpt")
    p.add_argument("--dump_preds", type=int, default=0, help="If >0, dump per-token predictions for the first N sequences to JSONL")
    p.add_argument("--topk", type=int, default=5, help="Top-k to dump when --dump_preds > 0")

    args = p.parse_args()

    mw = ModelWrapper(
        model_id=args.model_id,
        device_map=args.device_map,
        attn_implementation=None,
    )

#     # Build loaders
#     # train_loader = build_loaders(args.dataset_path, mw.tokenizer, args.seq_len, args.batch_size)
#     train_loader = build_wikitext_loaders(
#         tokenizer=mw.tokenizer,
#             seq_len=50,
#             batch_size=args.batch_size,
#             split="train",
#             name="wikitext-103-raw-v1",  # or "wikitext-2-raw-v1"
#             streaming=False              # True for huge runs
#     )
#     print("Data Loaded")
#     # Lens configs
#     lens_cfg = LensConfig(
#         layers=args.layers,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         epochs=args.epochs,
#         l2_to_identity=args.l2_to_identity,
#         use_amp=args.use_amp,
#     )

#     device = mw.model.device if hasattr(mw.model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     lens = ReverseLexicalLens(
#         d_model=mw.d_model,
#         vocab_size=mw.vocab_size,
#         E=mw.E_weight.to(device),
#         layers=args.layers,
#         cfg=lens_cfg,
#     ).to(device)

#     trainer = ReverseLexicalLensTrainer(
#         model_wrap=mw,
#         lens=lens,
#         train_cfg=TrainConfig(
#             seq_len=args.seq_len,
#             epochs=args.epochs,
#             lr=args.lr,
#             weight_decay=args.weight_decay,
#             use_amp=args.use_amp,
#         ),
#     )
    
#     test_loader = build_wikitext_loaders(
#         tokenizer=mw.tokenizer,
#             seq_len=50,
#             batch_size=args.batch_size,
#             split="test",
#             name="wikitext-103-raw-v1",  # or "wikitext-2-raw-v1"
#             streaming=False              # True for huge runs
#     )
#     print("Starting training of Reverse Lexical Lens", flush=True)
#     trainer.train_epochs(train_loader, layers=args.layers, epochs=args.epochs)

#     # quick evaluation on the training loader for demo purposes
#     print("Evaluating (on test set)")
#     metrics = trainer.evaluate(test_loader, layers=args.layers)
#     print(json.dumps(metrics, indent=2))

#     # Save lens
#     os.makedirs(args.save_dir, exist_ok=True)
#     torch.save({
#         "state_dict": lens.state_dict(),
#         "config": asdict(lens_cfg),
#         "model_id": args.model_id,
#         "layers": args.layers,
#         "d_model": mw.d_model,
#         "vocab_size": mw.vocab_size,
#     }, os.path.join(args.save_dir, "lens.pt"))
#     print(f"Saved lens to {os.path.join(args.save_dir, 'lens.pt')}")
#     dump_layer_topk_structured(
#     loader=test_loader,
#     mw=mw,
#     lens=lens,
#     layers=args.layers,
#     topk=1,
#     limit=1,
#     out_path=os.path.join(args.save_dir, "layer_top1.jsonl"),
# )


# ---- utilities for dumping predictions ----
@torch.no_grad()
def dump_layer_topk_structured(
    loader: DataLoader,
    mw: ModelWrapper,
    lens: ReverseLexicalLens,
    layers: List[int],
    topk: int,
    limit: int,
    out_path: str,
):
    """
    Write a single, human-readable JSON file with a clear schema.

    Schema:
    {
      "model_id": str,
      "vocab_size": int,
      "layers": [int, ...],
      "topk": int,
      "samples": [
        {
          "positions": [int, ...],           # non-PAD positions kept
          "input_tokens": [str, ...],        # tokens at those positions
          "input_ids": [int, ...],           # token ids at those positions
          "predictions": {
            "<layer>": {
              "top1_tokens": [str, ...],     # Top-1 token per position
              "top1_ids": [int, ...],        # Top-1 id per position
              "topk_tokens": [[str,...], ...],
              "topk_ids": [[int,...], ...],
              "topk_scores": [[float,...], ...]  # raw logits/similarities
            },
            ...
          }
        },
        ...
      ]
    }
    """
    import json, os
    dev = next(lens.parameters()).device
    pad_id = mw.tokenizer.pad_token_id

    samples = []
    seen = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(dev)
        hidden_states, _ = mw.forward_hidden_states(input_ids=input_ids)
        B, T = input_ids.shape

        for b in range(B):
            if seen >= limit:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model_id": mw.model_id,
                            "vocab_size": mw.vocab_size,
                            "layers": layers,
                            "topk": topk,
                            "samples": samples,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                return

            ids = input_ids[b].tolist()
            positions = [i for i, t in enumerate(ids) if t != pad_id]
            input_tokens = [mw.tokenizer.convert_ids_to_tokens(ids[i]) for i in positions]
            input_ids_kept = [ids[i] for i in positions]

            pred_dict = {}
            for l in layers:
                h = hidden_states[l][b:b+1]            # [1, T, d]
                logits = lens.forward_layer(h, l)[0]   # [T, V]
                vals, idx = logits.topk(topk, dim=-1)  # each: [T, topk]

                top1_tokens, top1_ids = [], []
                tk_tokens, tk_ids, tk_scores = [], [], []
                for t in positions:
                    pred_ids = [int(i) for i in idx[t].tolist()]
                    pred_toks = [mw.tokenizer.convert_ids_to_tokens(i) for i in pred_ids]
                    scores = [float(v) for v in vals[t].tolist()]

                    top1_tokens.append(pred_toks[0])
                    top1_ids.append(pred_ids[0])
                    tk_tokens.append(pred_toks)
                    tk_ids.append(pred_ids)
                    tk_scores.append(scores)

                pred_dict[str(l)] = {
                    "top1_tokens": top1_tokens,
                    "top1_ids": top1_ids,
                    "topk_tokens": tk_tokens,
                    "topk_ids": tk_ids,
                    "topk_scores": tk_scores,
                }

            samples.append(
                {
                    "positions": positions,
                    "input_tokens": input_tokens,
                    "input_ids": input_ids_kept,
                    "predictions": pred_dict,
                }
            )
            seen += 1

    # flush any remainder if limit not reached inside loop
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": mw.model_id,
                "vocab_size": mw.vocab_size,
                "layers": layers,
                "topk": topk,
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
