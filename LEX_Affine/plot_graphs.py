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
  "CE@layer_1": 0.7606056451201438,
  "Top1@layer_1": 0.9972000000000016,
  "Top5@layer_1": 0.9998500000000001,
  "Top10@layer_1": 0.9999300000000001,
  "CE@layer_2": 0.8448733745217323,
  "Top1@layer_2": 0.9962600000000021,
  "Top5@layer_2": 0.9998000000000001,
  "Top10@layer_2": 0.99992,
  "CE@layer_3": 1.0727963289022446,
  "Top1@layer_3": 0.9872900000000054,
  "Top5@layer_3": 0.9979800000000014,
  "Top10@layer_3": 0.9992200000000004,
  "CE@layer_4": 1.072502832174301,
  "Top1@layer_4": 0.9854700000000053,
  "Top5@layer_4": 0.998560000000001,
  "Top10@layer_4": 0.9993900000000004,
  "CE@layer_5": 1.1888123956918717,
  "Top1@layer_5": 0.9777300000000039,
  "Top5@layer_5": 0.9954900000000028,
  "Top10@layer_5": 0.9976000000000016,
  "CE@layer_6": 1.2645727190971374,
  "Top1@layer_6": 0.9702500000000028,
  "Top5@layer_6": 0.9914600000000041,
  "Top10@layer_6": 0.9946200000000031,
  "CE@layer_7": 1.2578001557588576,
  "Top1@layer_7": 0.9614099999999997,
  "Top5@layer_7": 0.9855200000000041,
  "Top10@layer_7": 0.9904200000000041,
  "CE@layer_8": 1.3902494766712188,
  "Top1@layer_8": 0.9528900000000001,
  "Top5@layer_8": 0.9803700000000034,
  "Top10@layer_8": 0.9850800000000037,
  "CE@layer_9": 1.5852682681083679,
  "Top1@layer_9": 0.9375999999999995,
  "Top5@layer_9": 0.9735400000000031,
  "Top10@layer_9": 0.9785400000000033,
  "CE@layer_10": 1.627046682715416,
  "Top1@layer_10": 0.9319699999999992,
  "Top5@layer_10": 0.9696500000000029,
  "Top10@layer_10": 0.9753300000000027,
  "CE@layer_11": 1.8476319735050202,
  "Top1@layer_11": 0.9116000000000001,
  "Top5@layer_11": 0.9606600000000023,
  "Top10@layer_11": 0.9681300000000032,
  "CE@layer_12": 1.8650901634693147,
  "Top1@layer_12": 0.9053900000000005,
  "Top5@layer_12": 0.9585600000000013,
  "Top10@layer_12": 0.9667100000000031,
  "CE@layer_13": 1.9583121120929718,
  "Top1@layer_13": 0.8965500000000007,
  "Top5@layer_13": 0.9540600000000005,
  "Top10@layer_13": 0.9634500000000014,
  "CE@layer_14": 2.2353912885189056,
  "Top1@layer_14": 0.8577700000000006,
  "Top5@layer_14": 0.931969999999999,
  "Top10@layer_14": 0.9478799999999995,
  "CE@layer_15": 2.458119762182236,
  "Top1@layer_15": 0.8208900000000006,
  "Top5@layer_15": 0.9095099999999994,
  "Top10@layer_15": 0.9281799999999997,
  "CE@layer_16": 2.6347192902565,
  "Top1@layer_16": 0.7924099999999994,
  "Top5@layer_16": 0.8931399999999995,
  "Top10@layer_16": 0.9150900000000008,
  "CE@layer_17": 2.708774146080017,
  "Top1@layer_17": 0.7851700000000004,
  "Top5@layer_17": 0.8882799999999993,
  "Top10@layer_17": 0.9113600000000003,
  "CE@layer_18": 2.804026447057724,
  "Top1@layer_18": 0.7770999999999996,
  "Top5@layer_18": 0.8833899999999998,
  "Top10@layer_18": 0.9078999999999999,
  "CE@layer_19": 2.895841549396515,
  "Top1@layer_19": 0.7615999999999996,
  "Top5@layer_19": 0.8751000000000005,
  "Top10@layer_19": 0.9019299999999993,
  "CE@layer_20": 2.9174164638519287,
  "Top1@layer_20": 0.7582600000000003,
  "Top5@layer_20": 0.8717200000000005,
  "Top10@layer_20": 0.8995499999999994,
  "CE@layer_21": 3.1550907711982727,
  "Top1@layer_21": 0.7001500000000002,
  "Top5@layer_21": 0.8271200000000004,
  "Top10@layer_21": 0.8614500000000003,
  "CE@layer_22": 3.319312904834747,
  "Top1@layer_22": 0.669019999999999,
  "Top5@layer_22": 0.7992299999999997,
  "Top10@layer_22": 0.8367800000000006,
  "CE@layer_23": 3.5325916156768797,
  "Top1@layer_23": 0.6176599999999995,
  "Top5@layer_23": 0.7459199999999988,
  "Top10@layer_23": 0.7893800000000001,
  "CE@layer_24": 3.529850008010864,
  "Top1@layer_24": 0.62689,
  "Top5@layer_24": 0.75547,
  "Top10@layer_24": 0.7989000000000006,
  "CE@layer_25": 3.361777566432953,
  "Top1@layer_25": 0.6539400000000006,
  "Top5@layer_25": 0.7834399999999994,
  "Top10@layer_25": 0.8222000000000004,
  "CE@layer_26": 3.0538427376747133,
  "Top1@layer_26": 0.7097499999999992,
  "Top5@layer_26": 0.8317600000000002,
  "Top10@layer_26": 0.8636000000000004,
  "CE@layer_27": 2.78104656624794,
  "Top1@layer_27": 0.7618999999999999,
  "Top5@layer_27": 0.8729100000000006,
  "Top10@layer_27": 0.8976099999999991,
  "CE@layer_28": 2.486730759859085,
  "Top1@layer_28": 0.81129,
  "Top5@layer_28": 0.9066699999999992,
  "Top10@layer_28": 0.9264300000000005,
  "CE@layer_29": 2.4132561562061308,
  "Top1@layer_29": 0.8165500000000002,
  "Top5@layer_29": 0.9117299999999999,
  "Top10@layer_29": 0.9307499999999995,
  "CE@layer_30": 2.363180060863495,
  "Top1@layer_30": 0.8380400000000001,
  "Top5@layer_30": 0.9229799999999996,
  "Top10@layer_30": 0.939809999999999,
  "CE@layer_31": 2.269478354215622,
  "Top1@layer_31": 0.8462499999999998,
  "Top5@layer_31": 0.9299599999999988,
  "Top10@layer_31": 0.9449999999999987,
  "CE@layer_32": 2.1296278784275056,
  "Top1@layer_32": 0.8609599999999995,
  "Top5@layer_32": 0.9395699999999999,
  "Top10@layer_32": 0.95249,
  "CE@layer_33": 2.133907390832901,
  "Top1@layer_33": 0.8621799999999998,
  "Top5@layer_33": 0.9413899999999991,
  "Top10@layer_33": 0.9538799999999995,
  "CE@layer_34": 2.135559295415878,
  "Top1@layer_34": 0.8613299999999999,
  "Top5@layer_34": 0.9421999999999993,
  "Top10@layer_34": 0.9543499999999995,
  "CE@layer_35": 1.9804722969532014,
  "Top1@layer_35": 0.8682000000000005,
  "Top5@layer_35": 0.9474699999999986,
  "Top10@layer_35": 0.9586500000000002,
  "CE@layer_36": 1.8497893785238266,
  "Top1@layer_36": 0.8972600000000003,
  "Top5@layer_36": 0.9576700000000006,
  "Top10@layer_36": 0.9664600000000021,
  "CE@layer_37": 1.876478118300438,
  "Top1@layer_37": 0.89568,
  "Top5@layer_37": 0.9570400000000007,
  "Top10@layer_37": 0.9656200000000023,
  "CE@layer_38": 1.8956321328878403,
  "Top1@layer_38": 0.8948600000000002,
  "Top5@layer_38": 0.9570400000000003,
  "Top10@layer_38": 0.9658300000000023,
  "CE@layer_39": 1.933395319223404,
  "Top1@layer_39": 0.8925299999999996,
  "Top5@layer_39": 0.9560500000000001,
  "Top10@layer_39": 0.9650600000000026,
  "CE@layer_40": 1.9675935020446778,
  "Top1@layer_40": 0.8890199999999995,
  "Top5@layer_40": 0.9536999999999999,
  "Top10@layer_40": 0.9631200000000018,
  "CE@layer_41": 2.0196523830890656,
  "Top1@layer_41": 0.88538,
  "Top5@layer_41": 0.9520699999999999,
  "Top10@layer_41": 0.9618400000000012,
  "CE@layer_42": 1.9665598590373994,
  "Top1@layer_42": 0.9043300000000003,
  "Top5@layer_42": 0.95872,
  "Top10@layer_42": 0.9673000000000026,
  "CE@layer_43": 2.044635115146637,
  "Top1@layer_43": 0.8989200000000005,
  "Top5@layer_43": 0.9574500000000001,
  "Top10@layer_43": 0.9665700000000026,
  "CE@layer_44": 2.102306825876236,
  "Top1@layer_44": 0.8945100000000004,
  "Top5@layer_44": 0.9566100000000002,
  "Top10@layer_44": 0.9663200000000022,
  "CE@layer_45": 2.194412759542465,
  "Top1@layer_45": 0.8885500000000003,
  "Top5@layer_45": 0.9537500000000001,
  "Top10@layer_45": 0.9647500000000027,
  "CE@layer_46": 2.2793660423755644,
  "Top1@layer_46": 0.8801699999999998,
  "Top5@layer_46": 0.9501099999999993,
  "Top10@layer_46": 0.9626800000000016,
  "CE@layer_47": 2.3511291580200195,
  "Top1@layer_47": 0.8690199999999998,
  "Top5@layer_47": 0.9451199999999987,
  "Top10@layer_47": 0.9586300000000003
}
# 1) From an in-memory dict:
# metrics = {...}  # the dict printed by trainer.evaluate(...)
# plot_ce_and_top1(metrics)
plot_single_series(metrics, "CE",title="Gemma-12b-it")
plot_single_series(metrics, "Top1",title="Gemma-12b-it")

# 2) From a saved JSON file:
# with open("metrics.json", "r") as f:
#     metrics = json.load(f)
# plot_ce_and_top1(metrics, title="Run A — CE & Top1 by Layer")
