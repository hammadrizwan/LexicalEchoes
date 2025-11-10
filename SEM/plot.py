import re
import matplotlib.pyplot as plt

def plot_sem_acc_by_layer(acc_json: dict, out_path: str):
    """
    Plot SEM probe accuracy vs. layer from a dict like:
      {"ACC(normL2)@layer_1": 0.12, "ACC(normL2)@layer_2": 0.15, ...}
    and save the figure to `out_path` (e.g., "sem_layers.png").

    Args:
        acc_json: dict mapping "ACC@layer_<n>" -> float accuracy
        out_path: filepath to save the plot (png/pdf/svg, etc.)
    """


    # Extract (layer, acc) pairs and sort by layer number
    pairs = []
    pat = re.compile(r"ACC_both@layer_(\d+)$")
    for k, v in acc_json.items():
        m = pat.match(k)
        if m:
            pairs.append((int(m.group(1)), float(v)))
    if not pairs:
        raise ValueError("No keys matching 'ACC@layer_<n>' found in acc_json.")

    pairs.sort(key=lambda x: x[0])
    layers = [p[0] for p in pairs]
    accs   = [p[1] for p in pairs]

    # Find best layer for annotation
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    best_layer, best_acc = layers[best_idx], accs[best_idx]

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(layers, accs, marker="o")
    plt.title("SEM Probe Accuracy by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.6)

    # Annotate best layer
    plt.scatter([best_layer], [best_acc])
    plt.annotate(
        f"best L{best_layer}: {best_acc:.3f}",
        xy=(best_layer, best_acc),
        xytext=(best_layer, best_acc + (max(accs) - min(accs)) * 0.05 if len(accs) > 1 else best_acc + 0.02),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

metrics={
  "ACC_anchor@layer_1": 0.9600638977635783,
  "ACC_para@layer_1": 0.9978035143769968,
  "ACC_both@layer_1": 0.9588658146964856,
  "mean_d_ap@layer_1": 0.09495624307150277,
  "mean_d_an@layer_1": 0.17995189282650384,
  "mean_d_pn@layer_1": 0.19861458184810493,
  "ACC_anchor@layer_2": 0.8258785942492013,
  "ACC_para@layer_2": 0.9920127795527156,
  "ACC_both@layer_2": 0.8218849840255591,
  "mean_d_ap@layer_2": 0.014228636595292594,
  "mean_d_an@layer_2": 0.01946181904512663,
  "mean_d_pn@layer_2": 0.023772017405436822,
  "ACC_anchor@layer_3": 0.7903354632587859,
  "ACC_para@layer_3": 0.9906150159744409,
  "ACC_both@layer_3": 0.7861421725239617,
  "mean_d_ap@layer_3": 0.019549706373542263,
  "mean_d_an@layer_3": 0.025448400283250183,
  "mean_d_pn@layer_3": 0.03150458882649105,
  "ACC_anchor@layer_4": 0.7891373801916933,
  "ACC_para@layer_4": 0.9876198083067093,
  "ACC_both@layer_4": 0.7833466453674122,
  "mean_d_ap@layer_4": 0.025085623021990345,
  "mean_d_an@layer_4": 0.03319347292161026,
  "mean_d_pn@layer_4": 0.040795866007241194,
  "ACC_anchor@layer_5": 0.8404552715654952,
  "ACC_para@layer_5": 0.9838258785942492,
  "ACC_both@layer_5": 0.8360623003194888,
  "mean_d_ap@layer_5": 0.03235211024007287,
  "mean_d_an@layer_5": 0.04904754672901699,
  "mean_d_pn@layer_5": 0.057841575886018744,
  "ACC_anchor@layer_6": 0.8568290734824281,
  "ACC_para@layer_6": 0.9754392971246006,
  "ACC_both@layer_6": 0.8508386581469649,
  "mean_d_ap@layer_6": 0.03716369846258491,
  "mean_d_an@layer_6": 0.06257299430216082,
  "mean_d_pn@layer_6": 0.07194786698530657,
  "ACC_anchor@layer_7": 0.8855830670926518,
  "ACC_para@layer_7": 0.9812300319488818,
  "ACC_both@layer_7": 0.8815894568690096,
  "mean_d_ap@layer_7": 0.04233287293285417,
  "mean_d_an@layer_7": 0.08111349869364748,
  "mean_d_pn@layer_7": 0.09034421880500385,
  "ACC_anchor@layer_8": 0.8885782747603834,
  "ACC_para@layer_8": 0.9838258785942492,
  "ACC_both@layer_8": 0.8847843450479234,
  "mean_d_ap@layer_8": 0.04540465159204821,
  "mean_d_an@layer_8": 0.08769975271754371,
  "mean_d_pn@layer_8": 0.09715919529858488,
  "ACC_anchor@layer_9": 0.8833865814696485,
  "ACC_para@layer_9": 0.9840255591054313,
  "ACC_both@layer_9": 0.8791932907348243,
  "mean_d_ap@layer_9": 0.047839541952259625,
  "mean_d_an@layer_9": 0.09330398886919783,
  "mean_d_pn@layer_9": 0.10299511870351462,
  "ACC_anchor@layer_10": 0.8785942492012779,
  "ACC_para@layer_10": 0.9818290734824281,
  "ACC_both@layer_10": 0.8738019169329073,
  "mean_d_ap@layer_10": 0.048119916196781604,
  "mean_d_an@layer_10": 0.09303006548851062,
  "mean_d_pn@layer_10": 0.10277120780925782,
  "ACC_anchor@layer_11": 0.8779952076677316,
  "ACC_para@layer_11": 0.9788338658146964,
  "ACC_both@layer_11": 0.8728035143769968,
  "mean_d_ap@layer_11": 0.048402949417837134,
  "mean_d_an@layer_11": 0.09276352146753487,
  "mean_d_pn@layer_11": 0.10258371901873964,
  "ACC_anchor@layer_12": 0.875,
  "ACC_para@layer_12": 0.9804313099041534,
  "ACC_both@layer_12": 0.8710063897763578,
  "mean_d_ap@layer_12": 0.04990283320077692,
  "mean_d_an@layer_12": 0.09620319494900231,
  "mean_d_pn@layer_12": 0.10640430688477172,
  "ACC_anchor@layer_13": 0.8801916932907349,
  "ACC_para@layer_13": 0.9804313099041534,
  "ACC_both@layer_13": 0.8755990415335463,
  "mean_d_ap@layer_13": 0.05067730445069627,
  "mean_d_an@layer_13": 0.09721838818571438,
  "mean_d_pn@layer_13": 0.10762185501023984,
  "ACC_anchor@layer_14": 0.8813897763578274,
  "ACC_para@layer_14": 0.9824281150159745,
  "ACC_both@layer_14": 0.8775958466453674,
  "mean_d_ap@layer_14": 0.05293585087497014,
  "mean_d_an@layer_14": 0.10039025768875695,
  "mean_d_pn@layer_14": 0.11105695769143181,
  "ACC_anchor@layer_15": 0.8925718849840255,
  "ACC_para@layer_15": 0.9868210862619808,
  "ACC_both@layer_15": 0.8893769968051118,
  "mean_d_ap@layer_15": 0.05795473880327929,
  "mean_d_an@layer_15": 0.10896346898981557,
  "mean_d_pn@layer_15": 0.12038967498956016,
  "ACC_anchor@layer_16": 0.9061501597444089,
  "ACC_para@layer_16": 0.990814696485623,
  "ACC_both@layer_16": 0.9029552715654952,
  "mean_d_ap@layer_16": 0.061550914503324526,
  "mean_d_an@layer_16": 0.11626694589472426,
  "mean_d_pn@layer_16": 0.12811183946106,
  "ACC_anchor@layer_17": 0.9107428115015974,
  "ACC_para@layer_17": 0.9932108626198083,
  "ACC_both@layer_17": 0.9079472843450479,
  "mean_d_ap@layer_17": 0.06482702932847194,
  "mean_d_an@layer_17": 0.12262097943705111,
  "mean_d_pn@layer_17": 0.1347848465219854,
  "ACC_anchor@layer_18": 0.9179313099041534,
  "ACC_para@layer_18": 0.994408945686901,
  "ACC_both@layer_18": 0.915535143769968,
  "mean_d_ap@layer_18": 0.06684242763791602,
  "mean_d_an@layer_18": 0.12641360460759732,
  "mean_d_pn@layer_18": 0.13873343502941984,
  "ACC_anchor@layer_19": 0.9199281150159745,
  "ACC_para@layer_19": 0.9942092651757188,
  "ACC_both@layer_19": 0.917332268370607,
  "mean_d_ap@layer_19": 0.06881342248651928,
  "mean_d_an@layer_19": 0.12862465813898813,
  "mean_d_pn@layer_19": 0.14109255114016822,
  "ACC_anchor@layer_20": 0.9169329073482428,
  "ACC_para@layer_20": 0.9940095846645367,
  "ACC_both@layer_20": 0.9143370607028753,
  "mean_d_ap@layer_20": 0.07131990051260009,
  "mean_d_an@layer_20": 0.13148815427630092,
  "mean_d_pn@layer_20": 0.14414471825852562,
  "ACC_anchor@layer_21": 0.9123402555910544,
  "ACC_para@layer_21": 0.992611821086262,
  "ACC_both@layer_21": 0.9085463258785943,
  "mean_d_ap@layer_21": 0.07405030783325338,
  "mean_d_an@layer_21": 0.13428684335927993,
  "mean_d_pn@layer_21": 0.1471299790203,
  "ACC_anchor@layer_22": 0.9121405750798722,
  "ACC_para@layer_22": 0.9928115015974441,
  "ACC_both@layer_22": 0.9085463258785943,
  "mean_d_ap@layer_22": 0.07699691841063408,
  "mean_d_an@layer_22": 0.13898729385373693,
  "mean_d_pn@layer_22": 0.15206527029173061,
  "ACC_anchor@layer_23": 0.9101437699680511,
  "ACC_para@layer_23": 0.992611821086262,
  "ACC_both@layer_23": 0.9063498402555911,
  "mean_d_ap@layer_23": 0.07942738470892174,
  "mean_d_an@layer_23": 0.1415202647876054,
  "mean_d_pn@layer_23": 0.154852743775319,
  "ACC_anchor@layer_24": 0.9071485623003195,
  "ACC_para@layer_24": 0.9928115015974441,
  "ACC_both@layer_24": 0.9031549520766773,
  "mean_d_ap@layer_24": 0.08169271366093486,
  "mean_d_an@layer_24": 0.14393855312380927,
  "mean_d_pn@layer_24": 0.15750764205623358,
  "ACC_anchor@layer_25": 0.9089456869009584,
  "ACC_para@layer_25": 0.9928115015974441,
  "ACC_both@layer_25": 0.9051517571884984,
  "mean_d_ap@layer_25": 0.08303263145513809,
  "mean_d_an@layer_25": 0.1463300675486985,
  "mean_d_pn@layer_25": 0.15979673044559672,
  "ACC_anchor@layer_26": 0.9067492012779552,
  "ACC_para@layer_26": 0.9920127795527156,
  "ACC_both@layer_26": 0.9029552715654952,
  "mean_d_ap@layer_26": 0.08317451881238827,
  "mean_d_an@layer_26": 0.14675441734230937,
  "mean_d_pn@layer_26": 0.16013551148743674,
  "ACC_anchor@layer_27": 0.9073482428115016,
  "ACC_para@layer_27": 0.9922124600638977,
  "ACC_both@layer_27": 0.9033546325878594,
  "mean_d_ap@layer_27": 0.08354148249656629,
  "mean_d_an@layer_27": 0.14841863815300763,
  "mean_d_pn@layer_27": 0.1616564533485772
}

plot_sem_acc_by_layer(metrics, "./sem_lens_ckpt//Llama-3.2-3B.png")