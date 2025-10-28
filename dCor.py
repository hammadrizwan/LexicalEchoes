# import os, json, re
# import numpy as np
# from plot_graph_counterfact import _to_layer_index, _read_last_jsonl, collect_layer_summaries
# # --- distance correlation (no SciPy needed) ---
# def _distance_correlation(x, y):
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
#     if x.size != y.size or x.size == 0:
#         return np.nan

#     # pairwise distance matrices
#     a = np.abs(x[:, None] - x[None, :])
#     b = np.abs(y[:, None] - y[None, :])

#     # double-centering
#     A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
#     B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

#     dcov2 = np.mean(A * B)
#     dvarx2 = np.mean(A * A)
#     dvary2 = np.mean(B * B)

#     # guard for tiny/negative due to fp error
#     dcov = np.sqrt(max(dcov2, 0.0))
#     dvarx = np.sqrt(max(dvarx2, 0.0))
#     dvary = np.sqrt(max(dvary2, 0.0))
#     if dvarx == 0.0 or dvary == 0.0:
#         return 0.0
#     return dcov / np.sqrt(dvarx * dvary)

# def compute_failure_entropy_dcor(
#     load_path,
#     entropy_json_path,
#     summary_filename="counterfact_summary.jsonl",
#     failure_key="Failure Rate",
#     skip_first_n=5,  # skip the first N layers
# ):
#     """
#     Reads per-layer summaries under load_path and a JSON {layer_index_str: entropy}.
#     Returns (dcor, aligned_layers, failure_vals, entropy_vals).
#     Ignores the first `skip_first_n` layers (after sorting numerically).
#     """
#     # 1) read entropy map
#     with open(entropy_json_path, "r") as f:
#         entropy_map = json.load(f)

#     # 2) read layer summaries
#     layers, summaries = collect_layer_summaries(load_path, summary_filename=summary_filename)

#     xs_failure, ys_entropy, keep_layers = [], [], []

#     for i, (name, s) in enumerate(zip(layers, summaries)):
#         idx = _to_layer_index(name)
#         if idx is np.inf:
#             continue
#         if i < skip_first_n:
#             continue  # skip early layers

#         # get failure rate
#         v = s.get(failure_key, None)
#         try:
#             v = float(v)
#         except Exception:
#             v = np.nan

#         # get entropy
#         e = entropy_map.get(str(int(idx)), None)
#         try:
#             e = float(e) if e is not None else np.nan
#         except Exception:
#             e = np.nan

#         if not (np.isnan(v) or np.isnan(e)):
#             xs_failure.append(v)
#             ys_entropy.append(e)
#             keep_layers.append(name)

#     if len(xs_failure) == 0:
#         print("[warn] No overlapping, valid values between Failure Rate and entropy file.")
#         return np.nan, [], [], []

#     dcor = _distance_correlation(np.array(xs_failure), np.array(ys_entropy))
#     print(f"[ok] dCor(Failure Rate, Entropy) over {len(xs_failure)} layers (skipped first {skip_first_n}) = {dcor:.6f}")
#     return dcor, keep_layers, xs_failure, ys_entropy

# paths=[
#         "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/Llama-3.2-3B-Instruct/_replace_att_averaging/",
#         "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/Llama-3.2-3B-Instruct/_replace_obj_averaging/",
#         "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/Llama-3.2-3B-Instruct/_replace_rel_averaging/",
#         "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/Llama-3.2-3B-Instruct/_swap_att_averaging/",
#         "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/Llama-3.2-3B-Instruct/_swap_obj_averaging/",

#         # "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/gemma-3-12b-it/_replace_att_averaging/",
#         # "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/gemma-3-12b-it/_replace_obj_averaging/",
#         # "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/gemma-3-12b-it/_replace_rel_averaging/",
#         # "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/gemma-3-12b-it/_swap_att_averaging/",
#         # "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/gemma-3-12b-it/_swap_obj_averaging/",
#        ]
# for path in paths:
#     splits=path.split("/")
#     model_name=splits[-3]
#     task=splits[-2]
#     print(model_name,task)
#     # continue
#     out_file_name=f"layer_summary_plot_{model_name}_{task}.png"
#     print(out_file_name)
#     dcor, layers_used, failure_vals, entropy_vals = compute_failure_entropy_dcor(
#             path,
#             "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/layer_by_layer/experiments/notebooks/Entropy_Llama3.2_Instruct_3B.json",
#             summary_filename="counterfact_summary.jsonl",   # change if your filename differs
#             failure_key="Failure Rate",
#         )
#     print("\n")
#     # print("out_file_name",out_file_name,"dcor", dcor)


import os, json, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr  # you'll need scipy

############################
# Existing helpers
############################

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
            line = f.readline()
            if not line:
                break
        f.seek(0)
    # The above early-read was a bug pattern in some snippets; let's do it properly:
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
    Returns:
        layer_names_sorted: list[str]  (layer folder names)
        summaries_sorted:   list[dict] (metrics for that layer)
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
    # sort numerically where possible
    entries.sort(key=lambda kv: _to_layer_index(kv[0]))
    layers = [kv[0] for kv in entries]
    summaries = [kv[1] for kv in entries]
    return layers, summaries

############################
# Layer value extraction
############################

def _extract_failure_rate_raw_order(layer_names, summaries, metric_key="Failure Rate"):
    """
    Produce parallel arrays: layer_indices, failure_rates
    layer_indices: numeric best-effort layer id
    failure_rates: float Failure Rate per layer (np.nan if missing)
    """
    layer_indices = []
    failure_rates = []
    for lname, s in zip(layer_names, summaries):
        idx = _to_layer_index(lname)

        v = s.get(metric_key, None)
        try:
            val = float(v) if v is not None else np.nan
        except Exception:
            val = np.nan

        layer_indices.append(idx)
        failure_rates.append(val)

    return layer_indices, np.array(failure_rates, dtype=float)

def load_layer_metric_json(metric_json_path):
    """
    Load external metric file:
      {"0": 0.0187, "1": 0.0674, "2": 0.0684, ...}

    Returns layer_ids (list[int]) and values (np.array[float]),
    sorted by layer id.
    """
    with open(metric_json_path, "r") as f:
        data = json.load(f)

    items = []
    for k, v in data.items():
        # parse layer id from key
        try:
            kid = int(k)
        except ValueError:
            m = re.search(r"(\d+)", k)
            if m:
                kid = int(m.group(1))
            else:
                continue
        try:
            val = float(v)
        except Exception:
            val = np.nan
        items.append((kid, val))

    items.sort(key=lambda x: x[0])
    layer_ids = [kid for (kid, _) in items]
    values = np.array([val for (_, val) in items], dtype=float)
    return layer_ids, values

def _align_by_layer_index(model_layer_ids, model_vals,
                          ext_layer_ids, ext_vals):
    """
    Intersect by numeric layer id.
    Keep only entries where both model and ext metric are finite.
    Return aligned arrays (model_vals_aligned, ext_vals_aligned, common_layer_ids)
    """
    model_map = {lid: val for lid, val in zip(model_layer_ids, model_vals)}
    ext_map   = {lid: val for lid, val in zip(ext_layer_ids, ext_vals)}

    common_ids = sorted(set(model_map.keys()) & set(ext_map.keys()))

    x_list = []
    y_list = []
    keep_ids = []
    for lid in common_ids:
        mv = model_map[lid]
        ev = ext_map[lid]
        if np.isfinite(mv) and np.isfinite(ev):
            x_list.append(mv)
            y_list.append(ev)
            keep_ids.append(lid)

    return np.array(x_list, dtype=float), np.array(y_list, dtype=float), keep_ids

############################
# Distance correlation (magnitude only)
############################

def _centered_distance_matrix(vec):
    """
    Compute double-centered distance matrix for vec (1D).
    """
    vec = np.asarray(vec, dtype=float).reshape(-1, 1)  # (n,1)
    A = np.abs(vec - vec.T)  # pairwise |xi - xj|
    row_mean = A.mean(axis=1, keepdims=True)
    col_mean = A.mean(axis=0, keepdims=True)
    grand_mean = A.mean()
    A_centered = A - row_mean - col_mean + grand_mean
    return A_centered

def distance_correlation(x, y):
    """
    Székely–Rizzo distance correlation.
    Always >= 0. Higher = stronger (possibly nonlinear) dependence.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = x.shape[0]
    if n < 2:
        return np.nan

    Ax = _centered_distance_matrix(x)
    Ay = _centered_distance_matrix(y)

    dCov2 = (Ax * Ay).mean()
    dVarX = (Ax * Ax).mean()
    dVarY = (Ay * Ay).mean()

    if dVarX <= 0 or dVarY <= 0:
        return np.nan

    dCor = np.sqrt(dCov2 / np.sqrt(dVarX * dVarY))
    return float(dCor)

############################
# Signed correlations
############################

def signed_correlations(x, y):
    """
    Compute Pearson r and Spearman rho (both signed).

    Returns dict:
      {
        "pearson_r": float,
        "pearson_p": float,
        "spearman_rho": float,
        "spearman_p": float
      }

    If not enough data or constant arrays, returns NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    # Pearson
    try:
        pr, pp = pearsonr(x, y)
    except Exception:
        pr, pp = (np.nan, np.nan)

    # Spearman (rank-based monotonic)
    try:
        sr, sp = spearmanr(x, y)
    except Exception:
        sr, sp = (np.nan, np.nan)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }

############################
# Master function
############################

def compute_model_metric_correlations(
    model_dir,
    metric_json_path,
    metric_key="Failure Rate",
    summary_filename="counterfact_summary.jsonl"
):
    """
    1. Pull model per-layer metric (default: Failure Rate).
    2. Pull external layer metric from JSON file.
    3. Align on shared layers.
    4. Compute:
       - Pearson r  (signed linear)
       - Spearman ρ (signed monotonic)
       - distance correlation (nonlinear magnitude only)

    Returns:
      results_dict, debug_dict

    results_dict:
      {
        "pearson_r": ...,
        "spearman_rho": ...,
        "distance_correlation": ...
      }

    debug_dict:
      {
        "common_layers": [...],
        "model_values_aligned": np.array([...]),
        "ext_values_aligned": np.array([...])
      }
    """

    # 1. get model failure rate (or other metric_key)
    layer_names, summaries = collect_layer_summaries(
        model_dir,
        summary_filename=summary_filename
    )
    model_layer_ids, model_vals = _extract_failure_rate_raw_order(
        layer_names,
        summaries,
        metric_key=metric_key
    )

    # 2. get external metric
    ext_layer_ids, ext_vals = load_layer_metric_json(metric_json_path)

    # 3. align them on layer id
    x_model, y_ext, common_ids = _align_by_layer_index(
        model_layer_ids, model_vals,
        ext_layer_ids, ext_vals
    )

    if x_model.size == 0 or y_ext.size == 0:
        print("[warn] no overlapping valid layers between model and external metric")
        return (
            {
                "pearson_r": np.nan,
                "spearman_rho": np.nan,
                "distance_correlation": np.nan,
            },
            {
                "common_layers": [],
                "model_values_aligned": x_model,
                "ext_values_aligned": y_ext,
            }
        )

    # 4a. signed correlations
    sig = signed_correlations(x_model, y_ext)

    # 4b. distance correlation magnitude
    dcor_val = distance_correlation(x_model, y_ext)

    results = {
        "pearson_r": sig["pearson_r"],
        "pearson_p": sig["pearson_p"],
        "spearman_rho": sig["spearman_rho"],
        "spearman_p": sig["spearman_p"],
        "distance_correlation": dcor_val,
    }

    debug = {
        "common_layers": common_ids,
        "model_values_aligned": x_model,
        "ext_values_aligned": y_ext,
    }

    return results, debug

####################################
# Example usage
####################################

# model_dir = "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-12b-pt_averaging/"
# info_metric_file = "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/dime_figures/gemma-3-12b-pt_dime_AP.json"


model_dir = "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/gemma-3-4b-pt_averaging/"
info_metric_file = "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/dime_figures/gemma-3-4b-pt_dime_AP.json"


# model_dir="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/counterfact_analysis/Llama-3.2-3B-Instruct_averaging/"
# info_metric_file="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/layer_by_layer/experiments/notebooks/DIME_Llama3.2_Instruct_3B.json"
results, dbg = compute_model_metric_correlations(
    model_dir=model_dir,
    metric_json_path=info_metric_file,
    metric_key="Failure Rate"
)

print("Pearson r:", results["pearson_r"])
print("Spearman rho:", results["spearman_rho"])
print("Distance Correlation:", results["distance_correlation"])
print("Layers used:", dbg["common_layers"])
