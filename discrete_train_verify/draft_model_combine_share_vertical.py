import os
import pickle

# from matplotlib import pyplot as plt
import numpy as np
import matplotlib as matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter

# cmap.to_rgba(list(np.arange(0,16)))
SMALL_SIZE = 12
BIGGERSIZE = 12 #255

params = {
    "axes.labelsize": BIGGERSIZE,
    "axes.titlesize": BIGGERSIZE,
    "font.size": BIGGERSIZE,
    "legend.fontsize": BIGGERSIZE,
    "ytick.labelsize": SMALL_SIZE,
    "xtick.labelsize": BIGGERSIZE,
    # "xtick.major.size": BIGGERSIZE,
    "axes.labelsize": SMALL_SIZE,
    # "axes.labelweight": BIGGERSIZE,
    "legend.title_fontsize": None,
}
matplotlib.rcParams.update(params)
matplotlib.use("pgf")
matplotlib.rc("pgf", texsystem="pdflatex")  # from running latex -v
preamble = matplotlib.rcParams.setdefault("pgf.preamble", [])
preamble += r"\usepackage{color}"
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#https://stackoverflow.com/questions/35091557/replace-nth-occurrence-of-substring-in-string
def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s

def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


def find_attr(attributes, keywords):
    for key in keywords:
        for attr in attributes:
            if key in attr[3]:
                # print("found", attr[3])
                return True

    return False

def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    elif x==0:
        return ".0"
    else:
        return "1.0"
major_formatter = FuncFormatter(my_formatter)

font = {"family": "serif", "weight": "bold", "size": BIGGERSIZE}

matplotlib.rc("font", **font)
SMALL_SIZE = 24
MEDIUM_SIZE = 32
# BIGGER_SIZE = 72

# plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
# plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
# plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
# plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
# plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# font_properties = {
#     "font.size": 72,
#     "font.weight": "bold",
#     "font.family": "serif",
# }
def flatten(l):
    return [item for sublist in l for item in sublist]
properties = {
    "size": BIGGERSIZE,
    "weight": "bold",
    "family": "serif",
}
counter = 0
trans = {
    "AGE": "age",
    "PTGENDER": "gender",
    "PTEDUCAT": "education",
    "PTETHCAT": "ethnicity",
    "PTRACCAT": "race",
    "PTMARRY": "marriage",
}
modes = ["disc"]
mode = modes[0]
models = ["logistic", "mlp3", "mlp6"]
dataset_splits = [0, 1]
# plt.rcParams["figure.figsize"] = [50.0,100.0]
plt.rcParams["figure.dpi"]=300.0
plt.rcParams["figure.autolayout"] = True
scaler = 13
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,sharey=True,figsize=(1*scaler,2*scaler),gridspec_kw={'width_ratios': [1,1, 1],'height_ratios': [1, 1],"wspace":0.05,"hspace":0.08})#,figsize=(16, 8))
axes = ((ax1, ax2, ax3), (ax4,ax5,ax6)) 
color_lst = list(mcolors.TABLEAU_COLORS.values())
color_lst += [mcolors.CSS4_COLORS["deeppink"],mcolors.CSS4_COLORS["gold"],mcolors.CSS4_COLORS["lawngreen"],mcolors.CSS4_COLORS["orangered"],mcolors.CSS4_COLORS["cyan"],mcolors.CSS4_COLORS["orange"]]
cmap = cm.ScalarMappable(cmap='rainbow')
# fig.subplots_adjust(hspace=0)                                                                                  
characters = ['a','b','c','d','e','f',]

for idx_dataset_split, dataset_split in enumerate(dataset_splits):
    for idx_model, model in enumerate(models):
        save_path = f"run7/model_{model}_split_{dataset_split}_log_{mode}"
        with open(save_path + ".pkl", "rb") as f:
            attrs_metrics = pickle.load(f)
        plots = dict()
        for attr, metrics in attrs_metrics.items():
            if find_attr(attr, ["PTEDUCAT", "AGE"]):
                continue
            if len(attr) not in plots:
                plots[len(attr)] = []
            plots[len(attr)].append((attr, metrics))

        # for leng, group in plots.items():
        group = flatten(list(plots.values()))
        leng = "all"
        plot_tmp = {
            "FPR": [],
            "FNR": [],
            "FNR_label": ["clean"],
            "FPR_label": ["clean"],
            "FNR_pvalue": [None],
            "FPR_pvalue": [None],
        }
        for measure in ["FPR", "FNR"]:
            # attr_tmp, std_acc_tmp, cer_acc_tmp, pgd_acc_tmp, p_values = [], [], [], [], []
            # plot_tmp[measure].append(metrics["clean"][measure])
            for idx, (attr, metrics) in enumerate(group):
                attr_ = str([tmp[3] for tmp in attr])
                for k, v in trans.items():
                    attr_ = attr_.replace(k, v)
                tmp_a = attr_.replace("[", "").replace("]", "").replace("'", "")
                if len(tmp_a.split(", "))==4:# True: #len(tmp_a.split(", ")) == 2 or len(tmp_a.split(", ")) == 3:
                    # tmp_a = tmp_a.replace(", ", ",\n")
                    tmp_a = nth_repl(tmp_a, ", ","\n",2)
                plot_tmp[measure + "_label"].append(tmp_a)
                if idx == 0:
                    plot_tmp[measure].append(metrics["clean"][measure])
                plot_tmp[measure].append(metrics["verify"][measure])
                mcmenar = metrics["verify"]["MN"]
                plot_tmp[measure + "_pvalue"].append([tmp2[1] for tmp2 in metrics["verify"]["MN"]])
            plot_tmp[measure] = (np.asarray(plot_tmp[measure]) * 1).tolist()
        # fig, ax = plt.subplots(figsize=(16, 8))
        ax = axes[idx_dataset_split][idx_model]
        # ax.axis("off")
        # ax.set_aspect(aspect=.2)
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax.yaxis.set_major_formatter(major_formatter)
        labels = []
        labels_global = []
        for a, b in zip(plot_tmp["FNR_label"], plot_tmp["FNR_pvalue"]):
            tmp1 = f"({np.mean(b):.3f})" if b != None else ""
            tmp2 = f"{a}"
            # labels.append(tmp2 + tmp1)
            labels.append(tmp1)
            labels_global.append(tmp2)

        x = np.arange(len(labels))
        width = 0.25
        width_ = 0.2

        c1 = ax.boxplot(
            plot_tmp["FPR"],
            positions=x - width / 2,
            sym="",
            widths=width_,
            vert=False,
            patch_artist=True,
            boxprops=dict(linewidth=2.5),
            # boxprops=dict(linestyle='-', linewidth=1.5),
            # flierprops=dict(linestyle='-', linewidth=1.5),
            # medianprops=dict(linestyle='-', linewidth=1.5),
            # whiskerprops=dict(linestyle='-', linewidth=1.5),
            # capprops=dict(linestyle='-', linewidth=1.5),
            # label="FPR",
            # widths=(1, 0.5, 1.2, 0.1),
        )
        c2 = ax.boxplot(
            plot_tmp["FNR"],
            positions=x + width / 2,
            sym="",
            widths=width_,
            vert=False,patch_artist=True,
            boxprops=dict(linewidth=2.5),
            # boxprops=dict(linestyle='-', linewidth=1.5),
            # flierprops=dict(linestyle='-', linewidth=1.5),
            # medianprops=dict(linestyle='-', linewidth=1.5),
            # whiskerprops=dict(linestyle='-', linewidth=1.5),
            # capprops=dict(linestyle='-', linewidth=1.5),
            # label = "FNR",
            # widths=(1, 0.5, 1.2, 0.1),
            # notch=True, bootstrap=10000,
        )
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.005)
        for i, (line1,line2) in enumerate(zip(c1['medians'],c2['medians'])):
            x1, y1 = line1.get_xydata()[0]
            x2, y2 = line2.get_xydata()[0]
            x = 0.5
            text = labels[i]
            # if idx_model ==2
            # if idx_model%2==0:
            #     x_ = np.maximum(0.2, x-0.2)
            # else:
            #     x_ = x
            ax.annotate(text, xy=(x, y1-width_-0.05))

        # colors = [
        #     "#D7191C",
        #     "#2C7BB6",
        # ]
        boxes = [c1, c2]
        ax.grid()
        # ax.set_facecolor("lightblue")
        ax.patch.set_facecolor('lightcyan')
        ax.patch.set_alpha(0.3)
        # for c_ in boxes:
        #     # for keyword in ["boxes","whiskers","caps","medians"]:
        #     for patch,patch_2,color_2 in zip(c_["boxes"],c_["medians"],color_lst):
        #         # set_box_color(patch,color_2)
        #         patch.set_color(color_2)
        #         patch_2.set_color(color_2)
        #     for patch,patch_2,color_2 in zip(c_["whiskers"],c_["caps"],np.repeat(color_lst,2)):
        #         patch.set_color(color_2)
        #         patch_2.set_color(color_2)
        # for patch in c1["boxes"]:
        #     patch.set(color="lightsteelblue",linewidth=2)
        for patch in c2["boxes"]:
            # change outline color
            patch.set(color='lightsteelblue', linewidth=2)
            # change fill color
            patch.set(facecolor = 'green' )
            # change hatch
            patch.set(hatch = '/')
            # set_box_color(c_, color)  # colors are from http://colorbrewer2.org/
        # for c_, color in zip(["FPR", "FNR"], colors):
        #     ax.plot([], c=color, label=c_)
        # ax.legend(fontsize=24)
        ax.set_xlabel(f"FPR / FNR", properties)

        if leng == 1:
            # ax.set_ylabel("Attribute Combinations and their (p-values)", properties)
            # ax.set_ylabel("p-values", properties)
            classes = f"{'NC vs MCI' if dataset_split==0 else 'MCI vs AD'}"
            title = f"{classes}, {model},  {'discrete' if mode=='disc' else 'continuous'}, {leng} combination"
        else:
            # ax.set_ylabel("Attribute Combinations and their (p-values)", properties)
            # ax.set_ylabel("p-values", properties)
            classes = f"{'NC vs MCI' if dataset_split==0 else 'MCI vs AD'}"
            title = f"{classes}, {model}"#,  {'discrete' if mode=='disc' else 'continuous'}, {leng} combinations"
        title = f"({characters[counter]}): "+title
        title = title.replace("logistic", "LRC")
        title = title.replace("mlp3", "MLP-3")
        title = title.replace("mlp6", "MLP-6")
        ax.set_title(title, properties)
        ticks = labels
        # if idx_model==0:

        ax.set_yticks(range(0, len(labels_global)), [tmp.replace("gender","sex") for tmp in labels_global])

        min = np.min(np.concatenate([plot_tmp["FPR"], plot_tmp["FNR"]]))
        max = np.max(np.concatenate([plot_tmp["FPR"], plot_tmp["FNR"]]))
        # min = 0.0 if min - 0.05 <= 0.0 else min
        # max = 1.0 if max + 0.05 >= 1.0 else max
        min = 0.0 #if min - 5.0 < 0.0 else min
        max = 1.0 #if max + 5 > 100.0 else max
        xticks = np.linspace(min, max, num=11)
        ax.set_xticks(xticks)

        # yticks = [f"{tmp:.1f}" for tmp in yticks]
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlim([0., 1.03])
        ax.xaxis.set_major_formatter(my_formatter)

        # print(counter + 1)
        counter += 1
'''
        fig.set_size_inches(16, 8)

'''
# ax4.set_xlabel("xaxis")                                                                                      
# ax1.set_ylabel("yaxis 1")                                                                                    
# ax4.set_ylabel("yaxis 2")   
# ncol=len(labels_global)//2,
# fig.legend(labels_global)
# handles_, labels_ = ax6.get_legend_handles_labels()
# labels_, handles_ = zip(*sorted(zip(labels_, handles_), key=lambda t: t[0]))
ax6.legend([c2["boxes"][0], c1["boxes"][0]], ['FNR', 'FPR'], loc='lower right')
# leg = fig.legend(labels_global, loc='lower center', bbox_to_anchor=(0.5,-0.08), ncol=len(labels_global), bbox_transform=fig.transFigure,fontsize = 8)
# for i, j in enumerate(leg.legendHandles):
    # j.set_color(color_lst[i])
fig.tight_layout()
# plt.yticks(rotation=-45)
prefix = "_"+os.getcwd().split("/")[-1].replace("health_fairness_","")
if os.path.isfile("year"+prefix+".pdf"):
    os.remove("year"+prefix+".pdf")
plt.savefig(
    # os.path.join("changed_plots_3y_models", f"{title.replace(', ','_')}.png"),
    "year"+prefix+".pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()
# print(os.path.join("changed_plots_3y_models", f"{title.replace(', ','_')}.png"))

