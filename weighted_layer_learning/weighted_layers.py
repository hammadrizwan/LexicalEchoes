# gemma12b_layer_weighted_triplet.py
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# =========================
# Token-wise mean pooling
# =========================
def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: [B, T, H]
    attention_mask:    [B, T] (1 for real tokens)
    returns:           [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
    return summed / denom


# =========================
# Sparse transforms
# =========================
def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax (Martins & Astudillo, 2016): projection onto the probability simplex
    producing exact zeros. Returns probs that sum to 1 (within numerical tolerance).
    """
    z = logits
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim=dim)

    r = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    # reshape r to broadcast along `dim`
    if dim != -1:
        view_shape = [1] * z.dim()
        view_shape[dim] = -1
        r = r.view(view_shape)

    support = 1 + r * z_sorted > z_cumsum  # boolean mask
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)  # [*, 1]
    tau = (z_cumsum.gather(dim, k - 1) - 1) / k.to(z.dtype)
    p = torch.clamp(z - tau, min=0.0)
    # (p should already sum to 1 on `dim`)
    s = p.sum(dim=dim, keepdim=True)
    p = p / (s + 1e-12)
    return p


def entmax15(logits: torch.Tensor, dim: int = -1, n_iter: int = 50, tol: float = 1e-6) -> torch.Tensor:
    """
    Entmax with alpha=1.5 (Peters et al., 2019). Produces sparse, smoother distributions than sparsemax.
    Simple Newton method; fine for tiny L (number of layers).
    """
    # shift for stability
    z = logits - logits.max(dim=dim, keepdim=True).values

    def _proj(z_in: torch.Tensor) -> torch.Tensor:
        # Solve for tau: p_i = relu(z_i - tau)^2 ; sum p_i = 1
        tau = torch.zeros_like(z_in.select(dim, 0).unsqueeze(dim))
        for _ in range(n_iter):
            relu = torch.clamp(z_in - tau, min=0.0)
            root = torch.sqrt(relu)
            g = root.sum(dim=dim, keepdim=True) - 1.0
            if g.abs().max() < tol:
                break
            denom = torch.clamp(relu, min=1e-12)
            dg = -0.5 * (denom.rsqrt()).sum(dim=dim, keepdim=True)
            tau = tau - g / torch.clamp(dg, min=-1e12, max=-1e-12)
        p = torch.clamp(z_in - tau, min=0.0).pow(2)
        s = p.sum(dim=dim, keepdim=True)
        p = p / (s + 1e-12)
        return p

    return _proj(z)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy (for optional sparsity encouragement with softmax/entmax)."""
    return -(p * (p.add(eps).log())).sum()


# =========================
# Backbone loader (handles encoder-only or decoder-only like Gemma 12B)
# =========================
def load_backbone(
    model_name: str,
    *,
    output_hidden_states: bool = True,
    torch_dtype=None,
    device_map: Optional[str] = None,
    gradient_checkpointing: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    cfg = AutoConfig.from_pretrained(model_name)
    common_kwargs = dict(
        output_hidden_states=output_hidden_states,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if load_in_8bit:
        common_kwargs.update(dict(load_in_8bit=True))
    if load_in_4bit:
        common_kwargs.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))

    if cfg.is_decoder and not cfg.is_encoder_decoder:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, **common_kwargs)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, cfg


# =========================
# Layer-weighted sentence encoder
# =========================
class LayerWeightedSentenceEncoder(nn.Module):
    """
    - pools mean over tokens at each layer
    - learns a sparse or dense distribution over layers (softmax / sparsemax / entmax15)
    - supports optional hard top-k straight-through gating
    - returns L2-normalized sentence embeddings
    """
    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-pt",
        use_hidden_states_only: bool = True,  # True -> exclude embedding layer
        dropout_prob: float = 0.0,
        proj_dim: Optional[int] = None,
        gradient_checkpointing: bool = True,
        # sparsity / weighting controls
        weight_transform: str = "sparsemax",  # "sparsemax" | "entmax15" | "softmax"
        topk: int = 0,                        # 0 disables hard top-k; >0 enables ST top-k
        learn_temperature: bool = False,
        torch_dtype=torch.bfloat16,
        device_map: Optional[str] = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.model, self.cfg = load_backbone(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            gradient_checkpointing=gradient_checkpointing,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        print(self.model)
        # figure out number of layers provided in hidden_states
        # hidden_states = [embeddings] + [layer1] + ... + [layerL]
        num_hidden = self.model.config.num_hidden_layers + 1
        self.layer_offset = 1 if use_hidden_states_only else 0
        self.num_layers = num_hidden - self.layer_offset

        # learned logits over layers; zeros -> uniform after normalization
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))

        # temperature
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("temperature", torch.tensor(1.0), persistent=False)

        self.weight_transform = weight_transform
        self.topk = int(topk)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.proj = None
        hidden_size = self.model.config.hidden_size
        if proj_dim is not None:
            self.proj = nn.Linear(hidden_size, proj_dim)

        # for logging/debug
        self._latest_weights_detached: Optional[torch.Tensor] = None

    def _normalize_weights(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature.clamp(min=1e-3)

        if self.weight_transform == "softmax":
            w = F.softmax(logits, dim=-1)
        elif self.weight_transform == "sparsemax":
            w = sparsemax(logits, dim=-1)
        elif self.weight_transform == "entmax15":
            w = entmax15(logits, dim=-1)
        else:
            raise ValueError(f"Unknown weight_transform: {self.weight_transform}")

        # optional hard top-k straight-through gating
        if self.topk and self.topk < w.numel():
            with torch.no_grad():
                hard = torch.zeros_like(w)
                topk_idx = torch.topk(w, k=self.topk, dim=-1).indices
                hard.scatter_(dim=-1, index=topk_idx, value=1.0)
                hard = hard / hard.sum(dim=-1, keepdim=True)
            w = (hard - w).detach() + w  # straight-through

        # cache a detached copy for logging
        self._latest_weights_detached = w.detach().cpu()
        return w

    def compute_layer_weights(self) -> torch.Tensor:
        """Return current layer distribution with gradients (useful for regularization)."""
        return self._normalize_weights(self.layer_logits)

    @torch.no_grad()
    def get_layer_weights(self) -> torch.Tensor:
        """Return current layer distribution (detached, on CPU) for logging."""
        if self._latest_weights_detached is not None:
            return self._latest_weights_detached.clone()
        w = self._normalize_weights(self.layer_logits)
        return w.detach().cpu()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden = outputs.hidden_states  # tuple of tensors

        # mean-pool each selected layer over tokens
        per_layer = torch.stack(
            [masked_mean_pool(all_hidden[li], attention_mask)
             for li in range(self.layer_offset, len(all_hidden))],
            dim=1
        )  # [B, L, H]

        # normalize layer weights
        weights = self.compute_layer_weights()  # [L]
        sent = torch.einsum("blh,l->bh", per_layer, weights)  # [B, H]

        if self.proj is not None:
            sent = self.proj(sent)

        sent = self.dropout(sent)
        # return L2-normalized embedding
        return F.normalize(sent, p=2, dim=-1)


# =========================
# Triplet data plumbing
# =========================
@dataclass
class TripletItem:
    anchor: str
    positive: str
    negative: str


class TripletTextDataset(Dataset):
    def __init__(self, triplets: List[TripletItem]):
        self.data = triplets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return {"anchor": t.anchor, "positive": t.positive, "negative": t.negative}


def make_collate_fn(tokenizer, max_length: int = 128):
    def collate(batch: List[Dict[str, str]]) -> Dict[str, Any]:
        a = [x["anchor"] for x in batch]
        p = [x["positive"] for x in batch]
        n = [x["negative"] for x in batch]
        toks = tokenizer(a + p + n, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        B = len(batch)
        anchor = {k: v[:B] for k, v in toks.items()}
        positive = {k: v[B:2 * B] for k, v in toks.items()}
        negative = {k: v[2 * B:3 * B] for k, v in toks.items()}
        return {"anchor": anchor, "positive": positive, "negative": negative}
    return collate


# =========================
# Loss: maximize cosine(a,p) - cosine(a,n)
# =========================
class CosineGapMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        pos = (a * p).sum(dim=-1)  # cosine (embeddings are L2-normalized)
        neg = (a * n).sum(dim=-1)
        gap = pos - neg
        return F.relu(self.margin - gap).mean()


# =========================
# Training config / loop
# =========================
@dataclass
class TrainConfig:
    model_name: str = "google/gemma-3-12b-pt"
    lr: float = 2e-4                 # you’re only training tiny params (layer logits + optional proj)
    weight_decay: float = 0.0
    batch_size: int = 16
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    max_length: int = 128
    margin: float = 0.3
    dropout_prob: float = 0.0
    proj_dim: Optional[int] = None
    gradient_checkpointing: bool = True
    fp16: bool = True
    num_workers: int = 2
    use_hidden_states_only: bool = True
    log_every: int = 50

    # sparsity / weighting
    weight_transform: str = "sparsemax"  # "sparsemax" | "entmax15" | "softmax"
    topk: int = 0                         # 0 for off; try 2-4 to hard-pick k layers
    entropy_lambda: float = 0.0           # e.g., 1e-3 if using softmax/entmax and want extra peakiness
    learn_temperature: bool = False

    # memory/precision options
    torch_dtype = torch.float32
    device_map: Optional[str] = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def train_triplet(
    train_triplets: List[TripletItem],
    cfg: TrainConfig
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    model = LayerWeightedSentenceEncoder(
        model_name=cfg.model_name,
        use_hidden_states_only=cfg.use_hidden_states_only,
        dropout_prob=cfg.dropout_prob,
        proj_dim=cfg.proj_dim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        weight_transform=cfg.weight_transform,
        topk=cfg.topk,
        learn_temperature=cfg.learn_temperature,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
    )

    # Freeze the massive backbone; train only layer weights (+ optional proj).
    for p in model.model.parameters():
        p.requires_grad = False

    # Ensure small head params require grad
    to_optimize = [p for p in model.parameters() if p.requires_grad]
    assert any(p.requires_grad for p in to_optimize), "No trainable params found."

    dataset = TripletTextDataset(train_triplets)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer, cfg.max_length),
        drop_last=False
    )

    optimizer = torch.optim.AdamW(to_optimize, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * math.ceil(len(dataset) / cfg.batch_size)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    loss_fn = CosineGapMarginLoss(margin=cfg.margin)

    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        for batch in loader:
            a = {k: v.to(device) for k, v in batch["anchor"].items()}
            p = {k: v.to(device) for k, v in batch["positive"].items()}
            n = {k: v.to(device) for k, v in batch["negative"].items()}

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                a_emb = model(**a)  # [B, D]
                p_emb = model(**p)
                n_emb = model(**n)
                loss = loss_fn(a_emb, p_emb, n_emb)

                # Optional entropy penalty on the current (with-grad) layer weights
                if cfg.entropy_lambda > 0.0:
                    w = model.compute_layer_weights()  # has grad
                    loss = loss + cfg.entropy_lambda * entropy(w)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(to_optimize, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % cfg.log_every == 0:
                w_det = model.get_layer_weights().numpy()
                print(
                    f"step {global_step:6d} | loss {loss.item():.4f} | "
                    f"layer_w sum={w_det.sum():.3f} max={w_det.max():.3f} "
                    f"nonzero={(w_det>1e-6).sum()}/{len(w_det)} | first5={w_det[:5]}"
                )

        print(f"[epoch {epoch+1}/{cfg.num_epochs}] done.")

    return model, tokenizer


# =========================
# Inference helper
# =========================
@torch.no_grad()
def encode_texts(
    texts: List[str],
    model: LayerWeightedSentenceEncoder,
    tokenizer,
    max_length: int = 128,
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> torch.Tensor:
    model.eval()
    if device is None:
        try:
            device = next(iter(model.parameters())).device
        except StopIteration:
            device = torch.device("cpu")

    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        toks = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        embs = model(**toks)  # [b, d] normalized
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


# =========================
# Example usage (toy)
# =========================
if __name__ == "__main__":
    # Replace with your real triplets
    toy = [
        TripletItem(
            anchor="A man is playing guitar.",
            positive="Someone plays an acoustic instrument.",
            negative="A dog is running in the park."
        ),
        TripletItem(
            anchor="Two kids are building a sandcastle.",
            positive="Children are making something on the beach.",
            negative="An adult is typing on a computer."
        ),
    ]

    cfg = TrainConfig(
        model_name="google/gemma-3-12b-pt",
        num_epochs=1,
        batch_size=8,
        margin=0.3,
        fp16=True,
        use_hidden_states_only=True,
        proj_dim=None,                 # set e.g. 256 to add a small projection
        dropout_prob=0.0,
        weight_transform="sparsemax",  # "sparsemax" | "entmax15" | "softmax"
        topk=0,                        # try 2-4 to hard-pick k layers
        entropy_lambda=0.0,            # use small value (e.g., 1e-3) if softmax/entmax
        gradient_checkpointing=True,
        # memory knobs:
        # torch_dtype=torch.bfloat16,    # good default on Ampere/Hopper
        device_map="auto",
        load_in_8bit=False,            # set True if using bitsandbytes 8-bit
        load_in_4bit=False,            # set True if using bitsandbytes 4-bit
    )

    # Train (backbone frozen; only layer weights/proj train)
    model, tokenizer = train_triplet(toy, cfg)

    # Encode a couple of texts and show cosine similarities
    texts = ["A person is playing music.", "A cat sleeps on the sofa.", "Kids are at the beach building something."]
    emb = encode_texts(texts, model, tokenizer)
    cos = (emb @ emb.t()).numpy()
    print("Cosine matrix:\n", cos)

    # Show learned layer distribution summary
    print("Layer weights (first 10):", model.get_layer_weights().numpy()[:10])
