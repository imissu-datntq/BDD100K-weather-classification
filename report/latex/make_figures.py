# Collect / draw the figures used by report/latex/main.tex into report/latex/figures/
# Run (from the repo root, after report/make_diagrams.py): .venv/Scripts/python.exe report/latex/make_figures.py
import os
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "figures")
DIAGRAMS = os.path.join(ROOT, "report", "figures")
RES = os.path.join(ROOT, "results")

MODELS = [("simple_cnn_noaug", "Simple CNN"), ("simple_cnn", "Simple CNN + Aug"),
          ("complex_cnn", "Complex CNN + Aug"), ("resnet50", "ResNet50 + Aug + Transfer")]
LOGS = {"simple_cnn_noaug": "simple_cnn_noaug_train.log", "simple_cnn": "logs/simple_cnn.log",
        "complex_cnn": "logs/complex_cnn.log", "resnet50": "logs/transfer.log"}
SHORT = ["D-C", "D-U", "N-C", "N-U", "DD-C", "DD-U"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def read_log(name):
    # per-epoch lines printed by the ValF1 callback in scripts/utils.py
    text = open(os.path.join(RES, LOGS[name]), encoding="utf-8", errors="ignore").read()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text.replace("\r", "\n"))
    pat = r"epoch (\d+): train loss ([\d.]+) acc ([\d.]+) \| val loss ([\d.]+) acc ([\d.]+) f1 ([\d.]+)"
    rows = [tuple(float(x) for x in m.groups()[1:]) for m in re.finditer(pat, text)]
    return pd.DataFrame(rows, columns=["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])


def learning_curves(path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for (name, label), c, mk in zip(MODELS, COLORS, MARKERS):
        h = read_log(name)
        print(name, "epochs", len(h), "best val F1", h["val_f1"].max(), "at epoch", h["val_f1"].idxmax() + 1)
        ep = np.arange(1, len(h) + 1)
        kw = dict(color=c, marker=mk, ms=3.5, lw=1.6, label=label)
        axes[0].plot(ep, h["val_f1"], **kw)
        axes[1].plot(ep, h["val_loss"], **kw)
        axes[2].plot(ep, h["train_acc"] - h["val_acc"], **kw)
    for ax, t in zip(axes, ["Validation macro-F1", "Validation loss (class-weighted)",
                            "Train accuracy - validation accuracy"]):
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e3e2de", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[2].axhline(0, color="#52514e", lw=0.8)
    axes[0].axvspan(0.5, 3.5, color="#f0efec", zorder=0)
    axes[0].text(2, 0.37, "ResNet50\nstage 1", ha="center", fontsize=8, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(path, dpi=200, facecolor="white")
    plt.close()


def confusion_matrices(path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
    for ax, (name, label) in zip(axes.flat, MODELS):
        p = pd.read_csv(os.path.join(RES, name + "_pred.csv"))
        cm = confusion_matrix(p["label"], p["pred"], labels=range(6), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(6), SHORT)
        ax.set_yticks(range(6), SHORT)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(label, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, facecolor="white")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in ["simple_cnn.png", "conv_block.png", "complex_cnn.png", "resnet50_transfer.png", "cnn_overview.png",
              "convolution.png", "max_pooling.png", "residual_block.png", "pipeline.png", "class_distribution.png"]:
        shutil.copy(os.path.join(DIAGRAMS, f), os.path.join(OUT, f))
    # photo grids are large PNGs, JPEG keeps the Overleaf upload small
    Image.open(os.path.join(RES, "samples.png")).convert("RGB").save(os.path.join(OUT, "samples.jpg"), quality=88)
    Image.open(os.path.join(RES, "complex_cnn_errors.png")).convert("RGB").save(os.path.join(OUT, "errors.jpg"),
                                                                                quality=85)
    learning_curves(os.path.join(OUT, "learning_curves.png"))
    confusion_matrices(os.path.join(OUT, "confusion_matrices.png"))
    print("figures in", OUT, sorted(os.listdir(OUT)))
