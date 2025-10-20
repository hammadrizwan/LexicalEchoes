import os, json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def print_tree(startpath, indent=""):
    for item in os.listdir(startpath):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}/")
            print_tree(path, indent + "    ")
        else:
            if("counterfact" in item):
                print(f"{indent}📄 {item}")
        
def collect_last_lines(root_dir):
    results = {}
    substrings = ["averaging", "lasttoken"]
    for subdir, _, files in os.walk(root_dir):
        # print(subdir)
        parse = ["lasttoken","averaging", "kalm", "e5","qwen","Promptriever"]
        if not any(word in subdir for word in parse):
            continue
        for file in files:
            if file == "counterfact_results.jsonl":
                file_path = os.path.join(subdir, file)

                # Read last line efficiently
                with open(file_path, "rb") as f:
                    f.seek(-2, os.SEEK_END)  # jump to the end
                    while f.read(1) != b"\n":
                        f.seek(-2, os.SEEK_CUR)
                    last_line = f.readline().decode()

                try:
                    data = json.loads(last_line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}")
                    continue

                # build key name from folder structure
                parts = os.path.normpath(subdir).split(os.sep)
                if len(parts) >= 2:
                    parent = parts[-2].capitalize()
                    current = parts[-1].replace(".", "").replace("_", "-").lstrip("-")
                    if any(sub in current for sub in substrings):
                        splits=current.split("-")
                        mode=splits[-1]
                        current="-".join(splits[:-1])
                        
                        if(mode =="lasttoken"):
                            parent=parent+"-LT"
                        else:
                            parent=parent+"-Avg"
                
                    key = f"{parent}_{current}"
                    # print(key)
         
                else:
                    key = parts[-1]

                results[key] = data

    return results

# Example usage
root_path = "/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/"
# print_tree(root_path)
data_dict = collect_last_lines(root_path)
# print(json.dumps(data_dict, indent=2))
final_dict={}
for k, v in data_dict.items():
    if("Gemma-3-4b" in k or "LT" in k ):
        continue
    # print(k.split("_")[0] ,k)
    v["model_name"] = k.split("_")[0]  # Extract model name from key
    
        
    v["model_task"] = "_".join(k.split("_")[1:]).lstrip("_") # Extract model task from key
    
    # print(v["model_name"],v["model_task"])

    final_dict[k]=v
# exit()

df = pd.DataFrame(final_dict).T # rows=model_task, cols=metrics
# print(df.head(3))
path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/Lexical_Semantic_Quantification/scpp_analysis/figures/"
# Make bar plots for each metric
# print(df.columns.tolist())
# Use a nicer theme
# Clean but strong style
# Identify columns
# import re
# non_metric_cols = {"model_name", "model_task"}
# metric_cols = [c for c in df.columns if c not in non_metric_cols]

# # Coerce metrics to numeric (in case they came in as strings)
# for c in metric_cols:
#     df[c] = pd.to_numeric(df[c], errors="coerce")

# # Normalize model names for consistent ordering (handles weird dashes/spaces)
# def norm(s):
#     s = str(s)
#     s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# df["model_name"] = df["model_name"].map(norm)
# df["model_task"] = df["model_task"].map(norm)

# # Explicit order: Gemma* then Llama* then the rest (alphabetical within each group)
# all_models = df["model_name"].dropna().unique().tolist()
# gemma = sorted([m for m in all_models if m.lower().startswith("gemma")])
# llama = sorted([m for m in all_models if m.lower().startswith("llama")])
# rest  = sorted([m for m in all_models if not (m.lower().startswith("gemma") or m.lower().startswith("llama"))])
# model_order = gemma + llama + rest
# df["model_name"] = pd.Categorical(df["model_name"], categories=model_order, ordered=True)

# # Stable order for tasks (datasets)
# task_order = sorted(df["model_task"].dropna().unique().tolist())
# df["model_task"] = pd.Categorical(df["model_task"], categories=task_order, ordered=True)

# # ---- Plot style ----
# sns.set_theme(style="whitegrid")
# palette = sns.color_palette("deep", n_colors=len(task_order))

# # ---- Grouped barplots (one figure per metric) ----
# for metric in metric_cols:
#     sub = df[["model_name", "model_task", metric]].dropna()

#     plt.figure(figsize=(12, 6))
#     ax = sns.barplot(
#         data=sub,
#         x="model_name",
#         y=metric,
#         hue="model_task",
#         order=model_order,
#         hue_order=task_order,
#         palette=palette,
#         dodge=True,
#         edgecolor="black",
#         linewidth=0.8
#     )

#     # Titles/labels
#     ax.set_title(metric, fontsize=16, fontweight="bold")
#     ax.set_xlabel("Model", fontsize=13)
#     ax.set_ylabel(metric, fontsize=13)
#     plt.xticks(rotation=30, ha="right")

#     # If metric seems bounded in [0,1], clamp nicely
#     vmax = sub[metric].max(skipna=True)
#     vmin = sub[metric].min(skipna=True)
#     if (vmin >= 0) and (vmax <= 1.05):
#         ax.set_ylim(0, 1)

#     # Legend outside for readability
#     ax.legend(title="Dataset (task)", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

#     sns.despine(trim=True)
#     plt.tight_layout()
#     plt.savefig(f"{path}{metric.replace(' ', '_')}_grouped_barplot.pdf", bbox_inches="tight")
#     plt.close()
#_________________________________________________________
# sns.set_theme(style="whitegrid")

# # # Use seaborn's "deep" palette (professional, strong colors)
# palette = sns.color_palette("deep", n_colors=df["model_name"].nunique())
# print(df["model_name"].values.tolist())
# # Define explicit order: first all Gemma, then Llama, then rest
# gemma = [m for m in df["model_name"].unique() if m.lower().startswith("gemma")]
# llama = [m for m in df["model_name"].unique() if m.lower().startswith("llama")]
# rest  = [m for m in df["model_name"].unique() if not (m.lower().startswith("gemma") or m.lower().startswith("llama"))]

# # Concatenate in the desired order
# model_order = gemma + llama + rest

# # Convert to categorical so seaborn respects the order
# df["model_name"] = pd.Categorical(df["model_name"], categories=model_order, ordered=True)

# for metric in df.columns:
#     if metric in ["model_task","model_name","Failure_severity_median_gap_Q4_minus_Q1"]:  # skip the grouping column
#         continue

#     plt.figure(figsize=(12,6))
#     ax = sns.boxplot(
#         data=df,
#         x="model_name",
#         y=metric,
#         order=model_order,
#         palette="deep",
#         width=0.5,
#         showmeans=True,
#         meanprops=dict(marker="o", markerfacecolor="black",
#                        markeredgecolor="black", markersize=6),
#         medianprops={"visible": False},           # hide median
#         boxprops=dict(edgecolor="black", linewidth=0.8),
#         whiskerprops=dict(color="black", linewidth=0.8),
#         capprops=dict(color="black", linewidth=0.8),
#         flierprops=dict(marker="o", markersize=3, alpha=0.5, color="black")
#     )

#     plt.xticks(rotation=30, ha="right")
#     plt.title(f"{metric} by Model", fontsize=16, fontweight="bold")
#     plt.ylabel(metric, fontsize=13)
#     plt.xlabel("Model", fontsize=13)

#     # Add headroom
#     y_min, y_max = ax.get_ylim()
#     plt.ylim(y_min, y_max * 1.1)

#     sns.despine(trim=True)
#     plt.tight_layout()
#     plt.savefig(f"{path}{metric.replace(' ', '_')}_whisker_mean.pdf", bbox_inches="tight")
#     plt.close()
# Order models explicitly: all Gemma*, then Llama*, then the rest
# --- Prep ---

# -------------------- Inputs --------------------
# Assumes you already have:
#   df  -> tidy table with columns: metric columns + ["model_name", "model_task"]
#   path -> directory to save figures (string, ends with "/" or use os.path.join)

# -------------------- Prep data --------------------
# -------------------- Prep data --------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import re

# -------------------- Inputs --------------------
# df: tidy dataframe with metric columns + ["model_name", "model_task"]
# path: output directory (string)

# -------------------- Prep data --------------------
df_plot = df.copy()

non_metric = {"model_name", "model_task"}
metric_cols = [c for c in df_plot.columns if c not in non_metric]

# Coerce metric columns to numeric
for c in metric_cols:
    df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

# Normalize names (unify dashes/spaces)
def norm(s):
    if s is None: return s
    s = str(s)
    s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s

df_plot["model_name"] = df_plot["model_name"].map(norm)
df_plot["model_task"] = df_plot["model_task"].map(norm)

# Explicit model order: Gemma* -> Llama* -> rest
all_models = df_plot["model_name"].dropna().unique().tolist()
gemma = sorted([m for m in all_models if m.lower().startswith("gemma")])
llama = sorted([m for m in all_models if m.lower().startswith("llama")])
rest  = sorted([m for m in all_models if not (m.lower().startswith("gemma") or m.lower().startswith("llama"))])
model_order = gemma + llama + rest
df_plot["model_name"] = pd.Categorical(df_plot["model_name"], categories=model_order, ordered=True)

# Stable task order (customize if desired)
task_order = sorted(df_plot["model_task"].dropna().unique().tolist())
df_plot["model_task"] = pd.Categorical(df_plot["model_task"], categories=task_order, ordered=True)

# -------------------- Style --------------------
sns.set_theme(style="whitegrid")
palette = sns.color_palette("deep", n_colors=len(model_order))

# Marker shapes for tasks (extend if needed)
marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
task_markers = {task: marker_pool[i % len(marker_pool)] for i, task in enumerate(task_order)}
marker_size = 22  # scatter size

# -------------------- Geometry --------------------
box_width = 0.70             # total width of the colored min–max box
inner_margin_rel = 0.06      # small inner margin so markers don't touch box edge
inner_margin = inner_margin_rel * box_width

# Metrics often bounded in [0,1]
bounded01 = {
    "Failure Rate",
    "Average Margin Violation",
    "FailRate_high_Q4",
    "FailRate_low_Q1",
    "FailRate_gap_Q4_minus_Q1",
    "AUC_LOcontrast_to_failure",
    "Failure_severity_median_gap_Q4_minus_Q1",
}
# -------------------- Plot loop --------------------
for metric in metric_cols:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-0.5, len(model_order) - 0.5)

    for i, model in enumerate(model_order):
        sub = df_plot.loc[df_plot["model_name"] == model, ["model_task", metric]].dropna()
        if sub.empty:
            continue

        vals = sub[metric].values.astype(float)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))

        # Colored min–max box (NO black border)
        rect = Rectangle(
            (i - box_width/2, vmin),
            box_width,
            max(vmax - vmin, 1e-12),
            facecolor=palette[i],
            edgecolor="none",   # <- removed black outline
            linewidth=0,
            alpha=0.35
        )
        ax.add_patch(rect)

        # Compute evenly spaced x-positions within the box for each task
        # positions are (k+1)/(n+1) across the usable width (with inner margins)
        n_tasks = len(task_order)
        usable_left  = rect.get_x() + inner_margin
        usable_right = rect.get_x() + rect.get_width() - inner_margin
        for k, task in enumerate(task_order):
            pos_rel = (k + 1) / (n_tasks + 1)               # 0..1 (e.g., 0.2, 0.4, 0.6, 0.8 for 4 tasks)
            x_pos   = usable_left + pos_rel * (usable_right - usable_left)

            # place the marker for this task IF present for this model
            y_vals = sub.loc[sub["model_task"] == task, metric].values
            if len(y_vals) == 0:
                continue
            y = float(y_vals[0])  # one value per task per model

            ax.scatter(
                x_pos, y,
                s=marker_size,
                marker=task_markers[task],
                color="black",
                zorder=3
            )

    # Legend: marker shape ↔ task
    # handles = [
    #     Line2D([0], [0], marker=task_markers[t], color="black", linestyle="",
    #            markersize=8, label=t)
    #     for t in task_order
    # ]
    # ax.legend(handles=handles, title="Task",
    #           loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # Axes cosmetics
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13
    })
    print("metric",metric)
    if(metric=="FailRate_low_Q1"):
        metric="LOS_Q1"
    if(metric=="FailRate_high_Q4"):
        metric="LOS_Q4"
    if(metric=="FailRate_gap_Q4_minus_Q1"):
        metric="LOS_Q4-Q1"
    if(metric=="AUC_LOcontrast_to_failure"):
        metric="AUC_LOcontrast"
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=11)
    ax.set_title(metric, fontsize=16, fontweight="bold")
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)

    if metric in bounded01:
        ax.set_ylim(0, 1)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{path}{metric.replace(' ', '_')}_rangebox_spread_markers.pdf", bbox_inches="tight")
    plt.close()

fig, ax = plt.subplots(figsize=(len(task_order) * 1.2, 0.6))
ax.axis("off")

handles = [
    Line2D([0], [0], marker=task_markers[t], color="black", linestyle="",
           markersize=10, label=t)
    for t in task_order
]

legend = ax.legend(
    handles=handles,
    ncol=len(task_order),
    loc="center",
    frameon=False,
    handletextpad=0.4,
    columnspacing=0.8,
    fontsize=13,
)

# Draw the canvas to get tight bounding box
fig.canvas.draw()
bbox = legend.get_window_extent()
bbox = bbox.transformed(fig.dpi_scale_trans.inverted())  # convert to inches

# Save *only* the legend area — no extra white space
fig.savefig(
    f"{path}task_legend_horizontal.pdf",
    bbox_inches=bbox,
    pad_inches=0.0,
    dpi=300,
)
plt.close()