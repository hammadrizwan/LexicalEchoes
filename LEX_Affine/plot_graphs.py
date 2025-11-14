import json
import re
import matplotlib.pyplot as plt

LAYER_RE = re.compile(r".+@layer_(\d+)")

def _extract_series(metrics: dict, prefix: str):
    """
    Return (layers, values) sorted by numeric layer index
    for keys like 'CE@layer_12' or 'Top1@layer_12'.
    """
    pairs = []
    for k, v in metrics.items():
        if not k.startswith(prefix + "@layer_"):
            continue
        m = LAYER_RE.match(k)
        if m:
            layer = int(m.group(1))
            pairs.append((layer, float(v)))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        raise ValueError(f"No keys found for prefix '{prefix}@layer_{{i}}' in metrics.")
    layers, vals = zip(*pairs)
    return list(layers), list(vals)

def plot_ce_and_top1(metrics: dict, title: str = "Reverse Lexical Lens — CE & Top1 by Layer"):
    layers_ce, ce_vals = _extract_series(metrics, "CE")
    layers_t1, t1_vals = _extract_series(metrics, "Top1")

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # CE (left y-axis)
    ln1 = ax1.plot(layers_ce, ce_vals, marker="o", linewidth=1.5, label="CE")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cross-Entropy (lower is better)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Top1 (right y-axis)
    ax2 = ax1.twinx()
    ln2 = ax2.plot(layers_t1, t1_vals, marker="s", linewidth=1.5, linestyle="--", label="Top1 Acc")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_ylim(0.0, 1.0)  # accuracy in [0,1]

    # One combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_single_series(metrics: dict, metric_prefix: str, title: str = None):
    """
    If you want just one plot (e.g., only CE or only Top1).
    metric_prefix: 'CE' or 'Top1' (also works for 'Top5', 'Top10', etc.)
    """
    layers, vals = _extract_series(metrics, metric_prefix)
    plt.figure(figsize=(8, 4))
    plt.plot(layers, vals, marker="o", linewidth=1.5)
    plt.xlabel("Layer")
    ylabel = {"CE": "Cross-Entropy", "Top1": "Top-1 Accuracy"}.get(metric_prefix, metric_prefix)
    plt.ylabel(ylabel)
    if metric_prefix.lower().startswith("top"):
        plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title(title or f"{metric_prefix} by Layer")
    plt.tight_layout()
    plt.savefig("./revlexlens_ckpt/"+title+"_"+metric_prefix+"_"+"_plot.png") 
    plt.show()

# --- Examples ---
metrics={
  "CE@layer_1": 0.12758748158961236,
  "Top1@layer_1": 0.9986526946107792,
  "Top5@layer_1": 1.0,
  "Top10@layer_1": 1.0,
  "CE@layer_2": 0.6597324310543532,
  "Top1@layer_2": 0.9715269461077891,
  "Top5@layer_2": 0.982764471057892,
  "Top10@layer_2": 0.984281437125756,
  "CE@layer_3": 0.9606921130311703,
  "Top1@layer_3": 0.9447604790419134,
  "Top5@layer_3": 0.9750000000000066,
  "Top10@layer_3": 0.9810878243513048,
  "CE@layer_4": 1.2963511086033728,
  "Top1@layer_4": 0.9144411177644713,
  "Top5@layer_4": 0.9554391217564853,
  "Top10@layer_4": 0.9657684630738546,
  "CE@layer_5": 1.6088850053246626,
  "Top1@layer_5": 0.8764071856287414,
  "Top5@layer_5": 0.9338323353293398,
  "Top10@layer_5": 0.9457185628742486,
  "CE@layer_6": 1.873231770273692,
  "Top1@layer_6": 0.8419960079840317,
  "Top5@layer_6": 0.9104990019960085,
  "Top10@layer_6": 0.9242814371257481,
  "CE@layer_7": 2.106022595645425,
  "Top1@layer_7": 0.8115668662674653,
  "Top5@layer_7": 0.8896706586826346,
  "Top10@layer_7": 0.9079141716566873,
  "CE@layer_8": 2.2735857885040924,
  "Top1@layer_8": 0.7903792415169661,
  "Top5@layer_8": 0.8773552894211578,
  "Top10@layer_8": 0.8965568862275455,
  "CE@layer_9": 2.3946421750767266,
  "Top1@layer_9": 0.7729141716566862,
  "Top5@layer_9": 0.8676347305389221,
  "Top10@layer_9": 0.8879041916167667,
  "CE@layer_10": 2.4812604688598725,
  "Top1@layer_10": 0.7607984031936129,
  "Top5@layer_10": 0.8602994011976055,
  "Top10@layer_10": 0.8832834331337331,
  "CE@layer_11": 2.516386207944143,
  "Top1@layer_11": 0.7556287425149701,
  "Top5@layer_11": 0.8564471057884226,
  "Top10@layer_11": 0.8803093812375256,
  "CE@layer_12": 2.504925365457516,
  "Top1@layer_12": 0.757035928143713,
  "Top5@layer_12": 0.8579540918163674,
  "Top10@layer_12": 0.8814770459081842,
  "CE@layer_13": 2.4072423787887938,
  "Top1@layer_13": 0.7744411177644703,
  "Top5@layer_13": 0.8713173652694608,
  "Top10@layer_13": 0.8948303393213574,
  "CE@layer_14": 2.108801642815748,
  "Top1@layer_14": 0.803323353293413,
  "Top5@layer_14": 0.8948502994011988,
  "Top10@layer_14": 0.9139920159680642,
  "CE@layer_15": 1.974522487370078,
  "Top1@layer_15": 0.8175748502994016,
  "Top5@layer_15": 0.9071556886227555,
  "Top10@layer_15": 0.9242415169660675,
  "CE@layer_16": 1.9128814305136066,
  "Top1@layer_16": 0.827894211576847,
  "Top5@layer_16": 0.9126646706586835,
  "Top10@layer_16": 0.9284431137724539,
  "CE@layer_17": 1.8923009324692444,
  "Top1@layer_17": 0.831786427145709,
  "Top5@layer_17": 0.9130239520958089,
  "Top10@layer_17": 0.927704590818363,
  "CE@layer_18": 1.8841684319064052,
  "Top1@layer_18": 0.8317165668662688,
  "Top5@layer_18": 0.9121856287425154,
  "Top10@layer_18": 0.926057884231537,
  "CE@layer_19": 1.8980745407873523,
  "Top1@layer_19": 0.8302295409181647,
  "Top5@layer_19": 0.9105389221556887,
  "Top10@layer_19": 0.9248403193612778,
  "CE@layer_20": 1.9683125126147698,
  "Top1@layer_20": 0.8253792415169664,
  "Top5@layer_20": 0.9077744510978046,
  "Top10@layer_20": 0.9229141716566871,
  "CE@layer_21": 2.059071253873631,
  "Top1@layer_21": 0.8175948103792411,
  "Top5@layer_21": 0.9039221556886228,
  "Top10@layer_21": 0.9197005988023961,
  "CE@layer_22": 2.134111285923484,
  "Top1@layer_22": 0.808842315369261,
  "Top5@layer_22": 0.8995808383233534,
  "Top10@layer_22": 0.9167465069860288,
  "CE@layer_23": 2.1928942943523504,
  "Top1@layer_23": 0.8024051896207582,
  "Top5@layer_23": 0.8957485029940123,
  "Top10@layer_23": 0.9142415169660678,
  "CE@layer_24": 2.266796255539991,
  "Top1@layer_24": 0.7932734530938119,
  "Top5@layer_24": 0.8898003992015966,
  "Top10@layer_24": 0.9102195608782435,
  "CE@layer_25": 2.348484411211071,
  "Top1@layer_25": 0.7854890219560882,
  "Top5@layer_25": 0.8857385229540916,
  "Top10@layer_25": 0.9068063872255493,
  "CE@layer_26": 2.4200462583534255,
  "Top1@layer_26": 0.772594810379242,
  "Top5@layer_26": 0.8769960079840313,
  "Top10@layer_26": 0.9002495009980044,
  "CE@layer_27": 2.5401185915141764,
  "Top1@layer_27": 0.753263473053892,
  "Top5@layer_27": 0.8639421157684632,
  "Top10@layer_27": 0.8900099800399196
}
# 1) From an in-memory dict:
# metrics = {...}  # the dict printed by trainer.evaluate(...)
# plot_ce_and_top1(metrics)
plot_single_series(metrics, "CE",title="Llama-3.2-3B")
plot_single_series(metrics, "Top1",title="Llama-3.2-3B")

# 2) From a saved JSON file:
# with open("metrics.json", "r") as f:
#     metrics = json.load(f)
# plot_ce_and_top1(metrics, title="Run A — CE & Top1 by Layer")
