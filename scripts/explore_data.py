# Thống kê dữ liệu cho mục II của báo cáo: số ảnh mỗi lớp, phân bố nhãn gốc, ảnh mẫu của từng lớp
# Chạy sau prepare_data.py: python scripts/explore_data.py

# 1. Import thư viện
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from prepare_data import ROOT, IMG_ROOT, CLASSES

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "results")
SEED = 42
N_PER_CLASS = 4  # số ảnh mẫu mỗi lớp


def image_path(row):
    # ưu tiên ảnh gốc 1280x720 cho đẹp, không có thì dùng ảnh đã resize 224x224
    raw = os.path.join(IMG_ROOT, "val", row["filename"])
    if os.path.exists(raw):
        return raw
    return os.path.join(DATA_DIR, "images", row["split_dir"], row["filename"])


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 2. Đọc 3 tập
    splits = {name: pd.read_csv(os.path.join(DATA_DIR, name + ".csv")) for name in ["train", "val", "test"]}

    # 3. Số ảnh mỗi lớp trong từng tập
    counts = pd.DataFrame({name: df["label"].value_counts() for name, df in splits.items()})
    counts = counts.reindex(range(len(CLASSES))).fillna(0).astype(int)
    counts.index = CLASSES
    counts.loc["total"] = counts.sum()
    print(counts)
    counts.to_csv(os.path.join(RESULT_DIR, "class_counts.csv"), index_label="class")

    # 4. Phân bố nhãn gốc timeofday x weather trên toàn bộ ảnh giữ lại
    all_df = pd.concat(splits.values(), ignore_index=True)
    attrs = pd.crosstab(all_df["timeofday"], all_df["weather"], margins=True, margins_name="total")
    print(attrs)
    attrs.to_csv(os.path.join(RESULT_DIR, "attribute_counts.csv"))

    # 5. Ảnh mẫu: mỗi lớp lấy ngẫu nhiên N_PER_CLASS ảnh trong tập test
    test = splits["test"]
    fig, axes = plt.subplots(len(CLASSES), N_PER_CLASS, figsize=(3.2 * N_PER_CLASS, 1.9 * len(CLASSES)))
    for c, name in enumerate(CLASSES):
        rows = test[test["label"] == c].sample(N_PER_CLASS, random_state=SEED)
        for j, (_, row) in enumerate(rows.iterrows()):
            ax = axes[c, j]
            ax.imshow(np.asarray(Image.open(image_path(row)).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(row["weather"], fontsize=8)  # nhãn thời tiết gốc trước khi gộp
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "samples.png"), dpi=150)
    plt.close()
    print("da luu", os.path.join(RESULT_DIR, "samples.png"))
