import os, json, re
import numpy as np
import matplotlib.pyplot as plt

def _to_layer_index(name):
    """best-effort to coerce a folder name to a sortable layer index"""
    try:
        return int(name)
    except ValueError:
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else np.inf

def _read_last_jsonl(path):
    """read the last non-empty JSON object from a .jsonl file; return None if not found"""
    if not os.path.isfile(path):
        return None
    last_obj = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last_obj = obj
            except Exception:
                continue
    return last_obj

def collect_layer_summaries(load_path, summary_filename="counterfact_summary.jsonl"):
    """
    Scan a model directory, find per-layer subdirs, read the last object
    in `counterfact_summary.jsonl` inside each, and return them sorted
    by numeric layer index when possible.
    Returns:
        layers_sorted: list[str]  (layer folder names)
        summaries_sorted: list[dict] (metrics for that layer)
    """
    entries = []
    for name in os.listdir(load_path):
        layer_dir = os.path.join(load_path, name)
        if not os.path.isdir(layer_dir):
            continue
        summary_path = os.path.join(layer_dir, summary_filename)
        data = _read_last_jsonl(summary_path)
        if data is None:
            continue
        entries.append((name, data))
    # Sort by numeric layer idx if present in folder name
    entries.sort(key=lambda kv: _to_layer_index(kv[0]))
    layers = [kv[0] for kv in entries]
    summaries = [kv[1] for kv in entries]
    return layers, summaries

def _extract_failure_rate_normalized_progress(layers, summaries, metric_key="Failure Rate"):
    """
    Convert per-layer metric into:
        progress_pct: np.array of [0..100] representing relative depth in the model
        y_values:     np.array of Failure Rate values (float or NaN)

    We don't return layer names anymore, because x-axis is now normalized % depth.
    """
    n_layers = len(layers)
    if n_layers == 0:
        return np.array([]), np.array([])

    if n_layers == 1:
        progress_pct = np.array([100.0], dtype=float)
    else:
        # even spacing from 0% to 100%
        progress_pct = np.linspace(0.0, 100.0, num=n_layers)

    y_values = []
    for s in summaries:
        v = s.get(metric_key, None)
        try:
            y_values.append(float(v) if v is not None else np.nan)
        except Exception:
            y_values.append(np.nan)

    return progress_pct, np.array(y_values, dtype=float)

def plot_failure_rate_multi(model_paths_dict,
                            save_path,
                            out_name="failure_rate_comparison_normdepth.png",
                            metric_key="Failure Rate",
                            title="Counterfact: Failure Rate vs Relative Layer Depth"):
    """
    model_paths_dict: { "Legend name for model": "/path/to/model_dir", ... }

    This produces ONE figure:
      x-axis = % depth through model (0 → 100)
      y-axis = Failure Rate (or other metric_key)
      each line = one model
    """

    plt.figure(figsize=(10,6))
    any_plotted = False

    for model_name, model_dir in model_paths_dict.items():
        layers, summaries = collect_layer_summaries(model_dir)
        if not layers:
            print(f"[warn] skipping {model_name}: no layer summaries found in {model_dir}")
            continue

        x_pct, y_vals = _extract_failure_rate_normalized_progress(
            layers, summaries, metric_key=metric_key
        )

        if x_pct.size == 0:
            print(f"[warn] skipping {model_name}: no usable data")
            continue

        # Draw the model's curve
        plt.plot(
            x_pct,
            y_vals,
            marker="o",
            label=model_name,
        )
        any_plotted = True

    if not any_plotted:
        print("[warn] nothing plotted (no valid models)")
        return None

    # Global styling
    plt.title(title)
    plt.xlabel("Relative layer depth (% of model)")
    plt.ylabel(metric_key)

    # Force 0→100% ticks and range
    tick_positions = [0, 20, 40, 60, 80, 100]
    plt.xticks(tick_positions, [f"{p}%" for p in tick_positions])
    plt.xlim(0, 100)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"[ok] Saved combined plot → {out_path}")
    return out_path



# ---------------- Example usage ----------------
# Pick ~3-4 model runs you want overlaid.
# Give them friendly legend names.
# paths=[
#     "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B-Instruct_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B-Instruct_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-it_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-it_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-it_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-it_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-1b-it_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-1b-it_lasttoken/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-pt_averaging/",
#        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-pt_lasttoken/"
#        ]

model_paths = {
    "Llama-3.2-3B-Instruct":
    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_averaging/", 
    "Llama-3.2-3B":
        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B_averaging/",
    
    "Gemma-3-4B-it":
        "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-it_averaging/",
    "Gemma-3-4B-pt":
    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-pt_averaging/",
    "Gemma-3-12B-it":
    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-it_averaging/",
    "Gemma-3-12b-pt":
    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-pt_averaging/",
    
}
metric_key="Average Margin Violation"
plot_failure_rate_multi(
    model_paths_dict=model_paths,
    save_path="./figures_counterfact/",
    out_name= "_".join(metric_key.split(" "))+"_comparison_combined_.png",
    metric_key=metric_key,
    title="Counterfact: Failure Rate vs Layer"
)
