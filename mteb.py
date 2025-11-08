import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from mteb import MTEB


class LayerEmbedder:
    """
    For a given transformer block index ℓ:
    - run model with output_hidden_states=True
    - take hidden_states[ℓ] (block output)
    - mean-pool across tokens using attention_mask
    Returns (batch, dim) embeddings.
    """
    def __init__(self, model, tokenizer, layer_idx, device="cuda", max_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.max_length = max_length

        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences, batch_size=16, **kwargs):
        all_vecs = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i:i+batch_size]

            toks = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(
                **toks,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = out.hidden_states
            h = hidden_states[self.layer_idx]  # (bsz, seq, dim)

            # mask-aware mean pool
            attn_mask = toks["attention_mask"].unsqueeze(-1)  # (bsz, seq, 1)
            summed = (h * attn_mask).sum(dim=1)               # (bsz, dim)
            lengths = attn_mask.sum(dim=1)                    # (bsz, 1)
            mean_pooled = summed / torch.clamp(lengths, min=1)

            all_vecs.append(mean_pooled.cpu())

        return torch.cat(all_vecs, dim=0).numpy()


class MTEBModelWrapper:
    """
    Thin wrapper so MTEB can call .encode() on this "model".
    """
    def __init__(self, layer_embedder, model_name, layer_idx, max_seq_length=256):
        self.layer_embedder = layer_embedder
        self._model_name = f"{model_name}-layer{layer_idx}"
        self.max_seq_length = max_seq_length

    def encode(self, sentences, **kwargs):
        bs = kwargs.get("batch_size", 16)
        return self.layer_embedder.encode(sentences, batch_size=bs)

    def get_sentence_embedding_dimension(self):
        test_vec = self.encode(["test"])
        return test_vec.shape[-1]

    @property
    def name(self):
        return self._model_name


def run_mteb_for_all_layers(
    hf_model_name: str,
    tasks_subset: list,
    output_root: str = "mteb_results",
    device: str = "cuda",
    device_map: str | dict = "auto",
    max_length: int = 256,
    dtype: torch.dtype = torch.float16,
):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=dtype,
        device_map=device_map
    )

    # ensure model on device for direct .to() calls in encode()
    # if device_map == "auto", huggingface will shard, so we don't .to(device)
    # but we still pass one logical device string into LayerEmbedder for .to() on inputs
    logical_device = device

    # probe hidden_states to know how many layers
    with torch.no_grad():
        dummy_inputs = tokenizer(
            ["hello world"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8,
        ).to(logical_device)
        dummy_out = model(
            **dummy_inputs,
            output_hidden_states=True,
            use_cache=False
        )
    num_hidden_states = len(dummy_out.hidden_states)
    layer_indices = list(range(num_hidden_states))

    # prep output dirs
    os.makedirs(output_root, exist_ok=True)
    model_root = os.path.join(output_root, hf_model_name.replace("/", "_"))
    os.makedirs(model_root, exist_ok=True)

    all_results = {}

    for layer_idx in layer_indices:
        print(f"\n=== Running MTEB for layer {layer_idx}/{num_hidden_states-1} ({hf_model_name}) ===")

        layer_embedder = LayerEmbedder(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=logical_device,
            max_length=max_length,
        )

        wrapped = MTEBModelWrapper(
            layer_embedder=layer_embedder,
            model_name=hf_model_name,
            layer_idx=layer_idx,
            max_seq_length=max_length,
        )

        evaluator = MTEB(tasks=tasks_subset)

        layer_outdir = os.path.join(model_root, f"layer_{layer_idx}")
        os.makedirs(layer_outdir, exist_ok=True)

        results = evaluator.run(
            wrapped,
            output_folder=layer_outdir
        )
        all_results[layer_idx] = results

        with open(os.path.join(layer_outdir, "results_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    with open(os.path.join(model_root, "all_layers_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    # Pick one model at a time and run this script.
    # Examples you said you want to cover:
    #   Llama 3.2 3B pretrained
    #   Llama 3.2 3B instruct
    #   Gemma 3 12B pretrained
    #   Gemma 3 12B instruction-tuned

    HF_MODEL_NAME = "meta-llama/Llama-3.2-3B"  # <-- change per run

    TASKS_SUBSET = [
        # paraphrase / similarity
        "STSBenchmark",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",

        # retrieval / reranking
        "MSMARCO",
        "FiQA2018",

        # duplicate / intent classification
        "StackOverflowDupQuestions",
        "Banking77Classification",
    ]

    OUTPUT_ROOT = "mteb_results"

    # For 3B on single GPU:
    DEVICE = "cuda"
    DEVICEMAP = "auto"  # "auto" is also fine for 3B
    DTYPE = torch.float16

    # For 12B on multi-GPU A100s you might want:
    # DEVICE = "cuda"
    # DEVICEMAP = "auto"
    # DTYPE = torch.bfloat16

    MAX_LENGTH = 256

    results = run_mteb_for_all_layers(
        hf_model_name=HF_MODEL_NAME,
        tasks_subset=TASKS_SUBSET,
        output_root=OUTPUT_ROOT,
        device=DEVICE,
        device_map=DEVICEMAP,
        max_length=MAX_LENGTH,
        dtype=DTYPE,
    )

    print("\nDone. Layers evaluated:", list(results.keys()))
