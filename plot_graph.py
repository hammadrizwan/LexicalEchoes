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
    """return (layers_sorted, summaries_sorted) where layers_sorted are numeric indices if possible"""
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
    # sort by numeric layer index when possible
    entries.sort(key=lambda kv: _to_layer_index(kv[0]))
    layers = [kv[0] for kv in entries]
    summaries = [kv[1] for kv in entries]
    return layers, summaries

def plot_layer_metrics(load_path, save_path,
                       metric_keys=("Failure Rate", "Average Margin Violation"),
                       out_name="layer_summary_plot.png",
                       title_prefix="Counterfact"):
    """
    Auto-discovers per-layer summaries and draws a line plot per metric.
    Saves to save_path/out_name and returns the path. Silently skips missing metrics.
    """
    layers, summaries = collect_layer_summaries(load_path)
    if not layers:
        print(f"[warn] No layer summaries found under: {load_path}")
        return None

    # x-axis: numeric layer indices if possible, else original order
    x_values = []
    x_labels = []
    for name in layers:
        idx = _to_layer_index(name)
        x_values.append(idx if idx is not np.inf else len(x_values))
        x_labels.append(name)

    # gather data per metric
    plotted_any = False
    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        print(key)
        y = []
        for s in summaries:
            v = s.get(key, None)
            # convert None/NaN to np.nan to allow masked plotting
            if v is None:
                y.append(np.nan)
            else:
                try:
                    y.append(float(v))
                except Exception:
                    y.append(np.nan)
        y = np.array(y, dtype=float)
        if np.all(np.isnan(y)):
            # nothing to plot for this metric
            continue
        # plot
        plt.plot(x_values, y, marker="o", label=key)
        plotted_any = True

    if not plotted_any:
        print(f"[warn] No valid metric values found for keys: {metric_keys}")
        return None

    plt.title(f"{title_prefix}: Per-Layer Summary")
    plt.xlabel("Layer")
    # show original folder names at ticks (more informative than pure indices)
    plt.xticks(x_values, x_labels, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_path, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"[ok] Saved plot → {out_path}")
    return out_path

# --------- Example usage ----------
# Assuming you already have args.save_path populated:
paths=[
    # "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B_lasttoken/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_lasttoken/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B_lasttoken/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B-Instruct_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-1B-Instruct_lasttoken/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-it_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-it_lasttoken/",
       "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-it_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-it_lasttoken/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-1b-it_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-1b-it_lasttoken/",
       "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-pt_averaging/",
    #    "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-pt_lasttoken/"
       ]
for path in paths:
    model_name=path.split("/")[-2]
    out_file_name=f"layer_summary_plot_{model_name}.png"
    print(out_file_name)
    plot_layer_metrics(path,
                       "./figures_counterfact/",
                       metric_keys=("Failure Rate","FailRate_low_Q1","FailRate_high_Q4","intrasims_dict"),
                       out_name=out_file_name,
                       title_prefix="Counterfact")
# plot_layer_metrics("/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_lasttoken",
#                    "./figures/",
#                    metric_keys=("Failure Rate","FailRate_low_Q1","FailRate_high_Q4"),
#                    out_name="layer_summary_plot_Llama-3.2-3B-Instruct-lt.png",
#                    title_prefix="Counterfact")
