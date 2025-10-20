import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz, minimum
__all__ = ["analyze_and_save_distances"]

# ----------------------- helpers -----------------------

def _to_clean_np(arr):
    x = np.asarray(list(arr), dtype=float)
    x = x[np.isfinite(x)]
    return x

def _scott_bandwidth(x):
    n = len(x)
    if n < 2:
        return 1.0
    s = np.std(x, ddof=1)
    if s == 0:
        return max(1e-3, 0.1 * (np.median(np.abs(x - np.median(x))) + 1e-6))
    return 1.06 * s * n ** (-1/5)

def _gaussian_kde_manual(x, grid, bw=None):
    x = np.asarray(x)
    if len(x) == 0:
        return np.zeros_like(grid)
    if bw is None:
        bw = _scott_bandwidth(x)
    denom = len(x) * bw * np.sqrt(2 * np.pi)
    z = (grid[:, None] - x[None, :]) / bw
    return np.sum(np.exp(-0.5 * z * z), axis=1) / denom

def _summary_stats(x):
    if len(x) == 0:
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }

# ---- ROUGE-n (F1) minimal implementation (token-based) ----
def _tokenize(s):
    return [t for t in str(s).lower().split() if t]

def _ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _rouge_n_f1(s1, s2, n=1):
    t1, t2 = _tokenize(s1), _tokenize(s2)
    g1, g2 = _ngrams(t1, n), _ngrams(t2, n)
    if not g1 or not g2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    from collections import Counter
    c1, c2 = Counter(g1), Counter(g2)
    overlap = sum((c1 & c2).values())
    p = overlap / max(1, sum(c1.values()))
    r = overlap / max(1, sum(c2.values()))
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}

def _avg_rouge(pairs, n):
    """pairs: list of (s1, s2)"""
    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ps = rs = fs = 0.0
    for s1, s2 in pairs:
        m = _rouge_n_f1(s1, s2, n=n)
        ps += m["precision"]; rs += m["recall"]; fs += m["f1"]
    k = len(pairs)
    return {"precision": ps/k, "recall": rs/k, "f1": fs/k}

# ---- threshold selection & confusion/flip ----
def _best_threshold(distances, labels):
    """
    distances: np.array of shape [N] (lower = more similar = predict positive if <= tau)
    labels:    np.array of {0,1}
    Returns tau*, confusion matrix at tau*, and error rate.
    """
    N = len(distances)
    if N == 0:
        return None, {"TP":0,"FP":0,"TN":0,"FN":0}, None

    # Candidates: midpoints between sorted unique distances, plus +/-inf guards
    order = np.argsort(distances)
    x = distances[order]
    y = labels[order]

    uniq = np.unique(x)
    if len(uniq) == 1:
        # Any threshold on that value — pick the value
        tau_candidates = [uniq[0]]
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        tau_candidates = [uniq[0]-1e-9] + list(mids) + [uniq[-1]+1e-9]

    best_err = float("inf"); best_tau = tau_candidates[0]; best_cm = None

    # cumulative positives/negatives to speed up counts
    # y_pred = (x <= tau)
    for tau in tau_candidates:
        pred = (distances <= tau).astype(int)
        TP = int(np.sum((pred == 1) & (labels == 1)))
        FP = int(np.sum((pred == 1) & (labels == 0)))
        TN = int(np.sum((pred == 0) & (labels == 0)))
        FN = int(np.sum((pred == 0) & (labels == 1)))
        err = (FP + FN) / N
        if err < best_err or (err == best_err and tau < best_tau):
            best_err = err
            best_tau = tau
            best_cm = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    return best_tau, best_cm, best_err

# ----------------------- main API -----------------------

def analyze_and_save_distances(
    incorrect_pairs,
    correct_pairs,
    title_prefix="Embedding distance",
    out_dir="dist_report",
    # NEW: custom group names
    group_neg_name="Non Pair Question",
    group_pos_name="Paired Question",
    # NEW: optional text pairs for ROUGE (each a list of (s1, s2))
    neg_text_pairs=None,
    pos_text_pairs=None,
    # NEW: styling hooks for plot colors/lines
    neg_color="#1f77b4",
    pos_color="#d62728",
    hist_alpha=0.35,
    kde_linewidth=2.0,
    ecdf_linewidth=2.0,
    tau_color="#2ca02c",
    tau_linestyle="--",
    tau_linewidth=1.5,
    violin_facealpha=0.35,
    box_facealpha=0.35,
):
    """
    Save figures and CSV summaries for two distance lists, plus extra metrics.

    Args:
      incorrect_pairs: iterable of floats for label==0 (non-paraphrase)
      correct_pairs:   iterable of floats for label==1 (paraphrase)
      group_neg_name:  name used for label==0 in outputs
      group_pos_name:  name used for label==1 in outputs
      neg_text_pairs:  optional list[(s1, s2)] for label==0 (for ROUGE stats)
      pos_text_pairs:  optional list[(s1, s2)] for label==1 (for ROUGE stats)
      neg_color, pos_color: colors for the two groups
      hist_alpha: alpha for histogram fill
      kde_linewidth, ecdf_linewidth: line widths
      tau_*: styling for vertical best-threshold line
      violin_facealpha, box_facealpha: fill transparency for violin/box plots

    Outputs in out_dir:
      - hist_kde.png, ecdf.png, violin.png, box.png
      - summary_stats.json/.csv
      - distances_long.csv
      - metrics.json (distance ratio, separation margin, best threshold, flip rate)
      - confusion_matrix.csv (at best threshold)
      - rouge_stats.json (if text pairs provided)
    Returns a dict of file paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = {}

    neg = _to_clean_np(incorrect_pairs)  # label 0
    pos = _to_clean_np(correct_pairs)    # label 1

    # ---- Save long-form CSV of distances ----
    # distances_csv = os.path.join(out_dir, "distances_long.csv")
    # with open(distances_csv, "w", newline="", encoding="utf-8") as f:
    #     w = csv.writer(f)
    #     w.writerow(["label", "group", "distance"])
    #     for v in neg:
    #         w.writerow([0, group_neg_name, float(v)])
    #     for v in pos:
    #         w.writerow([1, group_pos_name, float(v)])
    # files["distances_long_csv"] = distances_csv

    # ---- Summary stats ----
    stats = {
        group_neg_name: _summary_stats(neg),
        group_pos_name: _summary_stats(pos),
    }
    stats_json = os.path.join(out_dir, "summary_stats.json")
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    files["summary_stats_json"] = stats_json

    stats_csv = os.path.join(out_dir, "summary_stats.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "count", "mean", "std", "min", "p25", "median", "p75", "max"])
        for group, s in stats.items():
            row = [group] + [s.get(k, "") for k in ["count","mean","std","min","p25","median","p75","max"]]
            w.writerow(row)
    files["summary_stats_csv"] = stats_csv

    # If both empty, stop after CSVs
    if len(neg) == 0 and len(pos) == 0:
        return files

    # ---------------- METRICS ----------------

    # Distance Ratio & Separation Margin
    mean_neg = float(np.mean(neg)) if len(neg) else float("nan")
    mean_pos = float(np.mean(pos)) if len(pos) else float("nan")
    distance_ratio = (mean_pos / mean_neg) if (len(neg) and len(pos) and mean_neg != 0) else float("nan")
    separation_margin = (mean_neg - mean_pos) if (len(neg) and len(pos)) else float("nan")

    # Best threshold, confusion matrix, flip rate
    # Combine into one labeled array
    all_d = np.concatenate([neg, pos]) if len(neg) and len(pos) else (neg if len(neg) else pos)
    all_y = (np.concatenate([np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)])
             if (len(neg) and len(pos)) else
             (np.zeros(len(neg), dtype=int) if len(neg) else np.ones(len(pos), dtype=int)))

    tau_star, cm, flip_rate = _best_threshold(all_d, all_y)

    # Save confusion matrix CSV
    cm_csv = os.path.join(out_dir, "confusion_matrix.csv")
    with open(cm_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "Pred=1 (paraphrase)", "Pred=0 (non-paraphrase)"])
        w.writerow(["True=1 (paraphrase)", cm["TP"], cm["FN"]])
        w.writerow(["True=0 (non-paraphrase)", cm["FP"], cm["TN"]])
    files["confusion_matrix_csv"] = cm_csv
    # --- KDE overlap (area of min(f, g)) ---
    overlap_area = float("nan")
    overlap_percent = float("nan")
    if len(neg) and len(pos):
        # Build a shared grid (same logic as plots)
        all_vals = np.concatenate([neg, pos])
        lo, hi = np.min(all_vals), np.max(all_vals)
        pad = 0.05 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, 300)

        kde_neg = _gaussian_kde_manual(neg, grid)
        kde_pos = _gaussian_kde_manual(pos, grid)

        # Normalize to integrate to 1 on this grid
        from numpy import trapz, minimum
        area_neg = trapz(kde_neg, grid)
        area_pos = trapz(kde_pos, grid)
        if area_neg > 0 and area_pos > 0:
            kde_neg = kde_neg / area_neg
            kde_pos = kde_pos / area_pos
            overlap_area = float(trapz(minimum(kde_neg, kde_pos), grid))
            overlap_percent = float(overlap_area * 100.0)

    # Save metrics JSON
    metrics = {
        "group_names": {"neg": group_neg_name, "pos": group_pos_name},
        "mean_distance": {group_neg_name: mean_neg, group_pos_name: mean_pos},
        "distance_ratio_mean_pos_over_neg": distance_ratio,
        "separation_margin_mean_neg_minus_pos": separation_margin,
        "best_threshold_tau": tau_star,
        "flip_rate_at_best_tau": flip_rate,  # overall error rate at tau*
        "confusion_matrix_at_best_tau": cm,
        "decision_rule": "predict paraphrase if distance <= tau",
        "kde_overlap_area": overlap_area,         # 0..1
        "kde_overlap_percent": overlap_percent
    }
    metrics_json = os.path.join(out_dir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    files["metrics_json"] = metrics_json

    # ROUGE-1/2 (optional; only if texts provided)
    if neg_text_pairs is not None or pos_text_pairs is not None:
        r_stats = {}
        for name, pairs in [(group_neg_name, neg_text_pairs or []),
                            (group_pos_name, pos_text_pairs or [])]:
            r1 = _avg_rouge(pairs, n=1)
            r2 = _avg_rouge(pairs, n=2)
            r_stats[name] = {
                "rouge1": r1,
                "rouge2": r2,
                "count": len(pairs),
            }
        rouge_json = os.path.join(out_dir, "rouge_stats.json")
        with open(rouge_json, "w", encoding="utf-8") as f:
            json.dump(r_stats, f, indent=2)
        files["rouge_stats_json"] = rouge_json

    
    # ---------------- PLOTS (with custom colors) ----------------
    # Common grid for KDE/ECDF
    all_vals = np.concatenate([neg, pos]) if len(neg) and len(pos) else (neg if len(neg) else pos)
    lo, hi = np.min(all_vals), np.max(all_vals)
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 300)

    # Plot 1: Histogram + KDE
    plt.figure()
    if len(neg):
        plt.hist(neg, bins="auto", density=True, alpha=hist_alpha,
                 label=f"{group_neg_name} (hist)", color=neg_color)
        kde_neg = _gaussian_kde_manual(neg, grid)
        plt.plot(grid, kde_neg, linewidth=kde_linewidth,
                 label=f"{group_neg_name} (KDE)", color=neg_color)
    if len(pos):
        plt.hist(pos, bins="auto", density=True, alpha=hist_alpha,
                 label=f"{group_pos_name} (hist)", color=pos_color)
        kde_pos = _gaussian_kde_manual(pos, grid)
        plt.plot(grid, kde_pos, linewidth=kde_linewidth,
                 label=f"{group_pos_name} (KDE)", color=pos_color)
    if tau_star is not None:
        plt.axvline(tau_star, linestyle=tau_linestyle, linewidth=tau_linewidth,
                    color=tau_color, label=f"best τ = {tau_star:.4f}")
    if not np.isnan(overlap_percent):
        plt.plot([], [], label=f"Overlap percentage = {overlap_percent:.1f}%")
    if not np.isnan(overlap_area):
        plt.plot([], [], label=f"Overlap area = {overlap_area:.1f}%")

    plt.xlabel("L2 distance")
    plt.ylabel("Density")
    plt.title(f"{title_prefix}: Histogram + KDE")
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(out_dir, "hist_kde.png")
    plt.savefig(fig1, dpi=1000, bbox_inches="tight")
    plt.close()
    files["hist_kde_png"] = fig1

    # # Plot 2: ECDF
    # def ecdf(x):
    #     if len(x) == 0:
    #         return np.array([]), np.array([])
    #     xs = np.sort(x)
    #     ys = np.arange(1, len(xs) + 1) / len(xs)
    #     return xs, ys

    # plt.figure()
    # if len(neg):
    #     xs, ys = ecdf(neg)
    #     plt.plot(xs, ys, linewidth=ecdf_linewidth,
    #              label=f"{group_neg_name} (ECDF)", color=neg_color)
    # if len(pos):
    #     xs, ys = ecdf(pos)
    #     plt.plot(xs, ys, linewidth=ecdf_linewidth,
    #              label=f"{group_pos_name} (ECDF)", color=pos_color)
    # if tau_star is not None:
    #     plt.axvline(tau_star, linestyle=tau_linestyle, linewidth=tau_linewidth,
    #                 color=tau_color, label=f"best τ = {tau_star:.4f}")
    # plt.xlabel("L2 distance")
    # plt.ylabel("Empirical CDF")
    # plt.title(f"{title_prefix}: ECDF")
    # plt.legend()
    # plt.tight_layout()
    # fig2 = os.path.join(out_dir, "ecdf.png")
    # plt.savefig(fig2, dpi=1000, bbox_inches="tight")
    # plt.close()
    # files["ecdf_png"] = fig2

    # # Plot 3: Violin
    # data = []; labels = []
    # color_seq = []
    # if len(neg):
    #     data.append(neg); labels.append(f"{group_neg_name} (0)"); color_seq.append(neg_color)
    # if len(pos):
    #     data.append(pos); labels.append(f"{group_pos_name} (1)"); color_seq.append(pos_color)
    # if len(data):
    #     plt.figure()
    #     parts = plt.violinplot(data, showmeans=True, showextrema=True)

    #     # Color each violin body to match groups
    #     for body, c in zip(parts["bodies"], color_seq):
    #         body.set_facecolor(c)
    #         body.set_edgecolor(c)
    #         body.set_alpha(violin_facealpha)

    #     # Neutral styling for summary lines/extrema
    #     for key in ("cmeans", "cmedians", "cmaxes", "cmins", "cbars"):
    #         artist = parts.get(key, None)
    #         if artist is None:
    #             continue
    #         try:
    #             # Some are LineCollections, some are Line2D; both have set_color
    #             artist.set_color("#444444")
    #         except Exception:
    #             pass

    #     plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=0)
    #     plt.ylabel("L2 distance")
    #     plt.title(f"{title_prefix}: Violin")
    #     plt.tight_layout()
    #     fig3 = os.path.join(out_dir, "violin.png")
    #     plt.savefig(fig3, dpi=1000, bbox_inches="tight")
    #     plt.close()
    #     files["violin_png"] = fig3

    #     # Plot 4: Box
    #     plt.figure()
    #     bp = plt.boxplot(data, showmeans=True, patch_artist=True)

    #     # Face colors for boxes
    #     for patch, c in zip(bp["boxes"], color_seq):
    #         patch.set_facecolor(c)
    #         patch.set_alpha(box_facealpha)
    #         patch.set_edgecolor(c)

    #     # Neutral colors for whiskers/caps/medians/means
    #     for key in ("whiskers", "caps", "medians", "means"):
    #         for artist in bp[key]:
    #             try:
    #                 artist.set_color("#444444")
    #             except Exception:
    #                 pass

    #     plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=0)
    #     plt.ylabel("L2 distance")
    #     plt.title(f"{title_prefix}: Box")
    #     plt.tight_layout()
    #     fig4 = os.path.join(out_dir, "box.png")
    #     plt.savefig(fig4, dpi=1000, bbox_inches="tight")
    #     plt.close()
    #     files["box_png"] = fig4

    return files
