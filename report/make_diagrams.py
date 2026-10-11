# Draw the self-made figures of the report (theory illustrations, pipeline, model architectures,
# class distribution). Run: .venv/Scripts/python.exe report/make_diagrams.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
RESULTS = os.path.join(ROOT, "results")
DPI = 220

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
C_INPUT = "#f0efec"
C_AUG = "#fbd9c9"
C_CONV = "#86b6ef"
C_NORM = "#cde2fb"
C_POOL = "#3987e5"
C_FC = "#ffffff"
C_BLOCK = "#5598e7"
C_FROZEN = "#d9d8d4"
C_NEW = "#a8e0c9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color, fs=10, rot=0, weight="normal", ec=INK, lw=1.0):
    # (x, y) is the box centre
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=color, ec=ec, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, rotation=rot, weight=weight, color=INK)


def arrow(ax, p, q, color=INK, lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("saved", name)


def chain(ax, items, y, h, gap=0.35, x0=0.3, rot=90, fs=9.5):
    # draw tall narrow boxes left to right (GoogLeNet-style), return list of (x_left, x_right)
    x = x0
    spans = []
    for i, (text, color, w) in enumerate(items):
        box(ax, x + w / 2, y, w, h, text, color, fs=fs, rot=rot)
        spans.append((x, x + w))
        if i < len(items) - 1:
            arrow(ax, (x + w, y), (x + w + gap, y))
        x += w + gap
    return spans


# ----------------------------------------------------------------------------- theory figures
def fig_cnn_overview():
    fig, ax = canvas(18.2, 4.6, (11.4, 2.9))
    items = [("Input image\nH×W×3", C_INPUT, 1.9), ("Conv +\nReLU", C_CONV, 1.5), ("Pooling", C_POOL, 1.5),
             ("Conv +\nReLU", C_CONV, 1.5), ("Pooling", C_POOL, 1.5), ("Flatten /\nGAP", C_NORM, 1.4),
             ("Fully\nconnected", C_FC, 1.5), ("Softmax\n(K classes)", C_FC, 1.7)]
    spans = chain(ax, items, 2.7, 2.2, gap=0.45, rot=0, fs=9.5)
    ax.annotate("", xy=(spans[0][0], 0.95), xytext=(spans[4][1], 0.95),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.text((spans[0][0] + spans[4][1]) / 2, 0.45, "feature extraction (learned local patterns, shrinking resolution)",
            ha="center", fontsize=9, color=INK2)
    ax.annotate("", xy=(spans[5][0], 0.95), xytext=(spans[7][1], 0.95),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.text((spans[5][0] + spans[7][1]) / 2, 0.45, "classification head", ha="center", fontsize=9, color=INK2)
    save(fig, "cnn_overview.png")


def grid(ax, x0, y0, m, cell, colors=None, title=None, fs=11):
    # draw matrix m with its top-left corner at (x0, y0)
    n_r, n_c = m.shape
    for i in range(n_r):
        for j in range(n_c):
            c = colors[i][j] if colors is not None else "white"
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=c, ec=INK, lw=0.8))
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, f"{m[i, j]:g}", ha="center", va="center", fontsize=fs)
    if title:
        ax.text(x0 + n_c * cell / 2, y0 + 0.35, title, ha="center", fontsize=10, color=INK)


def fig_convolution():
    X = np.array([[1, 2, 0, 1, 3], [0, 1, 3, 2, 1], [2, 1, 0, 1, 0], [1, 3, 2, 0, 2], [0, 1, 1, 2, 1]])
    K = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    Y = np.array([[sum(X[i:i + 3, j:j + 3].ravel() * K.ravel()) for j in range(3)] for i in range(3)])
    fig, ax = canvas(13.2, 6.2, (9.5, 4.5))
    hi = "#cde2fb"
    cx = [[hi if i < 3 and j < 3 else "white" for j in range(5)] for i in range(5)]
    grid(ax, 0.2, 5.4, X, 0.9, cx, "Input X (5×5)")
    ax.text(5.15, 3.2, "★", ha="center", va="center", fontsize=20)
    grid(ax, 5.9, 4.5, K, 0.9, [[C_CONV] * 3] * 3, "Kernel W (3×3)")
    ax.text(9.15, 3.2, "=", ha="center", va="center", fontsize=22)
    cy = [[hi if (i, j) == (0, 0) else "white" for j in range(3)] for i in range(3)]
    grid(ax, 9.8, 4.5, Y, 0.9, cy, "Feature map Y (3×3)")
    ax.text(6.6, 0.25, "Y[0,0] = 1·1 + 2·0 + 0·(−1) + 0·1 + 1·0 + 3·(−1) + 2·1 + 1·0 + 0·(−1) = 0"
            "    (stride 1, no padding)", ha="center", fontsize=9, color=INK2)
    save(fig, "convolution.png")


def fig_pooling():
    X = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 3, 2], [2, 6, 1, 1]])
    q = [["#cde2fb", "#cde2fb", "#fbd9c9", "#fbd9c9"], ["#cde2fb", "#cde2fb", "#fbd9c9", "#fbd9c9"],
         ["#c9ecdc", "#c9ecdc", "#fbe7b3", "#fbe7b3"], ["#c9ecdc", "#c9ecdc", "#fbe7b3", "#fbe7b3"]]
    fig, ax = canvas(9.4, 4.6, (6.6, 3.2))
    grid(ax, 0.2, 4.0, X, 0.9, q, "Input (4×4)")
    arrow(ax, (4.1, 2.2), (5.6, 2.2))
    ax.text(4.85, 2.5, "2×2 max pool\nstride 2", ha="center", va="bottom", fontsize=9, color=INK2)
    Y = np.array([[4, 5], [6, 3]])
    grid(ax, 6.1, 3.1, Y, 0.9, [["#cde2fb", "#fbd9c9"], ["#c9ecdc", "#fbe7b3"]], "Output (2×2)")
    save(fig, "max_pooling.png")


def fig_residual():
    fig, ax = canvas(7.0, 8.0, (3.6, 4.1))
    box(ax, 3.0, 0.6, 2.2, 0.7, "x", C_INPUT)
    box(ax, 3.0, 2.0, 3.0, 0.8, "Weight layer", C_CONV)
    box(ax, 3.0, 3.3, 3.0, 0.7, "ReLU", C_NORM)
    box(ax, 3.0, 4.6, 3.0, 0.8, "Weight layer", C_CONV)
    ax.add_patch(plt.Circle((3.0, 6.0), 0.35, fc="white", ec=INK, lw=1))
    ax.text(3.0, 6.0, "+", ha="center", va="center", fontsize=14)
    box(ax, 3.0, 7.2, 3.0, 0.7, "ReLU", C_NORM)
    for a, b in [(0.95, 1.6), (2.4, 2.95), (3.65, 4.2), (5.0, 5.65), (6.35, 6.85)]:
        arrow(ax, (3.0, a), (3.0, b))
    ax.plot([3.0, 5.6, 5.6], [1.25, 1.25, 6.0], color=INK, lw=1.2)
    arrow(ax, (5.6, 6.0), (3.35, 6.0))
    ax.text(5.75, 3.6, "identity\nshortcut x", fontsize=9, color=INK2, va="center")
    ax.text(0.15, 3.3, "F(x)", fontsize=11, color=INK2, va="center")
    ax.text(3.45, 6.35, "F(x) + x", fontsize=9, color=INK2)
    save(fig, "residual_block.png")


# ----------------------------------------------------------------------------- practical figures
def fig_pipeline():
    fig, ax = canvas(16.0, 6.2, (11, 4.3))
    steps = [("BDD100K labels\ntimeofday, weather", C_INPUT), ("Merge into\n6 classes", C_NORM),
             ("Clean & split\ntrain / val / test", C_NORM), ("Resize 224×224\n+ normalize", C_NORM),
             ("Augmentation\n(train only)", C_AUG)]
    w, h, y = 2.6, 1.4, 5.0
    xs = [1.5 + i * 3.25 for i in range(len(steps))]
    for i, (t, c) in enumerate(steps):
        box(ax, xs[i], y, w, h, t, c, fs=9)
        if i:
            arrow(ax, (xs[i - 1] + w / 2, y), (xs[i] - w / 2, y))
    models = [("Model 1: Simple CNN", C_CONV), ("Model 2: Complex CNN", C_CONV),
              ("Model 3: ResNet50 transfer", C_CONV)]
    ym = [3.0, 2.0, 1.0]
    ax.plot([xs[-1], xs[-1]], [y - h / 2, 3.65], color=INK, lw=1.2)
    ax.plot([6.0, xs[-1]], [3.65, 3.65], color=INK, lw=1.2)
    ax.plot([6.0, 6.0], [3.65, 1.0], color=INK, lw=1.2)
    for (t, c), yy in zip(models, ym):
        arrow(ax, (6.0, yy), (6.6, yy))
        box(ax, 8.5, yy, 3.8, 0.75, t, c, fs=9)
        ax.plot([10.4, 11.2], [yy, yy], color=INK, lw=1.2)
    ax.plot([11.2, 11.2], [3.0, 1.0], color=INK, lw=1.2)
    arrow(ax, (11.2, 2.0), (11.9, 2.0))
    box(ax, 13.75, 2.0, 3.6, 1.6, "Evaluate on test set\naccuracy, precision,\nrecall, macro-F1,\nconfusion matrix", C_FC, fs=9)
    ax.text(8.5, 0.15, "training: class-weighted cross-entropy, checkpoint chosen by validation macro-F1",
            ha="center", fontsize=8.5, color=INK2)
    save(fig, "pipeline.png")


def fig_simple_cnn():
    fig, ax = canvas(21.2, 5.4, (11.9, 3.0))
    items = [("Image 224×224×3", C_INPUT, 0.95), ("Flip + Zoom (aug)", C_AUG, 0.95),
             ("3×3 Conv (32), ReLU", C_CONV, 0.95), ("2×2 MaxPool", C_POOL, 0.8),
             ("3×3 Conv (64), ReLU", C_CONV, 0.95), ("2×2 MaxPool", C_POOL, 0.8),
             ("3×3 Conv (128), ReLU", C_CONV, 0.95), ("2×2 MaxPool", C_POOL, 0.8),
             ("3×3 Conv (128), ReLU", C_CONV, 0.95), ("2×2 MaxPool", C_POOL, 0.8),
             ("Flatten (25088)", C_NORM, 0.95), ("FC (256), ReLU", C_FC, 0.95),
             ("Dropout 0.5", C_NORM, 0.8), ("FC (6), Softmax", C_FC, 0.95)]
    spans = chain(ax, items, 3.2, 3.6, gap=0.62, fs=9)
    shapes = {0: "224²×3", 2: "224²×32", 3: "112²×32", 4: "112²×64", 5: "56²×64", 6: "56²×128", 7: "28²×128",
              8: "28²×128", 9: "14²×128", 10: "25088", 11: "256", 13: "6"}
    for i, s in shapes.items():
        ax.text((spans[i][0] + spans[i][1]) / 2, 1.0, s, ha="center", fontsize=8, color=INK2)
    save(fig, "simple_cnn.png")


def fig_conv_block():
    xm, xs, xc = 3.4, 8.3, 5.85  # x of main branch, shortcut branch, centre line
    fig, ax = canvas(10.4, 10.6, (5.2, 5.4))
    box(ax, xc, 0.6, 4.0, 0.75, "Input (C_in channels)", C_INPUT, fs=9)
    main = [("3×3 Conv (C_out)", C_CONV), ("BatchNorm", C_NORM), ("ReLU", C_NORM),
            ("3×3 Conv (C_out)", C_CONV), ("BatchNorm", C_NORM)]
    ys = [2.0, 3.1, 4.1, 5.2, 6.3]
    for (t, c), y in zip(main, ys):
        box(ax, xm, y, 3.6, 0.7, t, c, fs=9)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (xm, a + 0.35), (xm, b - 0.35))
    box(ax, xs, 3.1, 3.3, 0.7, "1×1 Conv (C_out)", C_CONV, fs=9)
    box(ax, xs, 4.1, 3.0, 0.7, "BatchNorm", C_NORM, fs=9)
    arrow(ax, (xs, 3.45), (xs, 3.75))
    ax.plot([xc, xc], [0.975, 1.35], color=INK, lw=1.2)
    ax.plot([xm, xs], [1.35, 1.35], color=INK, lw=1.2)
    arrow(ax, (xm, 1.35), (xm, 1.65))
    arrow(ax, (xs, 1.35), (xs, 2.75))
    ax.add_patch(plt.Circle((xc, 7.5), 0.33, fc="white", ec=INK, lw=1))
    ax.text(xc, 7.5, "+", ha="center", va="center", fontsize=14)
    ax.plot([xm, xm], [6.65, 7.5], color=INK, lw=1.2)
    arrow(ax, (xm, 7.5), (xc - 0.33, 7.5))
    ax.plot([xs, xs], [4.45, 7.5], color=INK, lw=1.2)
    arrow(ax, (xs, 7.5), (xc + 0.33, 7.5))
    box(ax, xc, 8.6, 3.0, 0.7, "ReLU", C_NORM, fs=9)
    box(ax, xc, 9.75, 3.0, 0.7, "2×2 MaxPool", C_POOL, fs=9)
    arrow(ax, (xc, 7.83), (xc, 8.25))
    arrow(ax, (xc, 8.95), (xc, 9.4))
    ax.text(xs + 0.15, 2.0, "shortcut", ha="left", fontsize=8.5, color=INK2)
    ax.text(0.65, 4.1, "main\nbranch", ha="center", va="center", fontsize=8.5, color=INK2)
    save(fig, "conv_block.png")


def fig_complex_cnn():
    fig, ax = canvas(18.9, 5.4, (11.6, 3.3))
    items = [("Image 224×224×3", C_INPUT, 0.95), ("Flip + Zoom (aug)", C_AUG, 0.95),
             ("3×3 Conv (32), stride 2\nBN, ReLU", C_CONV, 1.05), ("ConvBlock (64)", C_BLOCK, 1.05),
             ("ConvBlock (128)", C_BLOCK, 1.05), ("ConvBlock (256)", C_BLOCK, 1.05),
             ("ConvBlock (512)", C_BLOCK, 1.05), ("Global AvgPool", C_POOL, 0.95), ("Dropout 0.3", C_NORM, 0.8),
             ("FC (256), ReLU", C_FC, 0.95), ("Dropout 0.3", C_NORM, 0.8), ("FC (6), Softmax", C_FC, 0.95)]
    spans = chain(ax, items, 3.2, 3.6, gap=0.62, fs=9)
    shapes = {0: "224²×3", 2: "112²×32", 3: "56²×64", 4: "28²×128", 5: "14²×256", 6: "7²×512", 7: "512", 9: "256", 11: "6"}
    for i, s in shapes.items():
        ax.text((spans[i][0] + spans[i][1]) / 2, 1.0, s, ha="center", fontsize=8, color=INK2)
    save(fig, "complex_cnn.png")


def fig_resnet_transfer():
    fig, ax = canvas(17.6, 7.2, (10.5, 4.3))
    items = [("Image 224×224×3", C_INPUT, 0.95), ("Flip + Zoom (aug)", C_AUG, 0.95),
             ("To Caffe input\n(RGB→BGR, −mean)", C_AUG, 0.95),
             ("conv1: 7×7 Conv (64) /2\n+ 3×3 MaxPool /2", C_FROZEN, 1.05),
             ("conv2_x\n3 bottlenecks (256)", C_FROZEN, 1.05), ("conv3_x\n4 bottlenecks (512)", C_FROZEN, 1.05),
             ("conv4_x\n6 bottlenecks (1024)", C_FROZEN, 1.05), ("conv5_x\n3 bottlenecks (2048)", C_FROZEN, 1.05),
             ("Global AvgPool", C_NEW, 0.95), ("Dropout 0.3", C_NEW, 0.8), ("FC (6), Softmax", C_NEW, 0.95)]
    spans = chain(ax, items, 4.6, 4.2, gap=0.62, fs=9)
    shapes = {0: "224²×3", 3: "56²×64", 4: "56²×256", 5: "28²×512", 6: "14²×1024", 7: "7²×2048", 8: "2048", 10: "6"}
    for i, s in shapes.items():
        ax.text((spans[i][0] + spans[i][1]) / 2, 2.15, s, ha="center", fontsize=8, color=INK2)

    def bracket(x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y + 0.25, y, y, y + 0.25], color=INK, lw=1)
        ax.text((x1 + x2) / 2, y - 0.35, text, ha="center", va="top", fontsize=8.5)
    bracket(spans[3][0], spans[7][1], 1.6, "Pretrained ResNet50 backbone (ImageNet, 23.6M params)\n"
                                          "Stage 1: frozen     Stage 2: unfrozen, fine-tuned (lr 1e-4)")
    bracket(spans[8][0], spans[10][1], 1.6, "New classification head\ntrained in both stages")
    save(fig, "resnet50_transfer.png")


def fig_class_distribution():
    path = os.path.join(RESULTS, "class_counts.csv")
    if os.path.exists(path):
        counts = pd.read_csv(path, index_col="class").drop(index="total")
    else:  # numbers printed by prepare_data.py
        counts = pd.DataFrame({"train": [13373, 10386, 18346, 3677, 2059, 1541],
                               "val": [3343, 2597, 4587, 919, 515, 385],
                               "test": [2402, 1862, 3279, 584, 402, 312]},
                              index=["daytime_clear", "daytime_unclear", "night_clear", "night_unclear",
                                     "dawndusk_clear", "dawndusk_unclear"])
    total = counts.sum(axis=1).sort_values()
    share = total / total.sum() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh(total.index, total.values, height=0.62, color=BLUE)
    for i, (v, s) in enumerate(zip(total.values, share.values)):
        ax.text(v + total.max() * 0.01, i, f"{v:,}  ({s:.1f}%)", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, total.max() * 1.22)
    ax.set_xlabel("number of images (train + val + test)")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_distribution.png"), dpi=DPI, facecolor="white")
    plt.close(fig)
    print("saved class_distribution.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_cnn_overview()
    fig_convolution()
    fig_pooling()
    fig_residual()
    fig_pipeline()
    fig_simple_cnn()
    fig_conv_block()
    fig_complex_cnn()
    fig_resnet_transfer()
    fig_class_distribution()
