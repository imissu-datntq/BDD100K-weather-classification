# Đánh giá lại các mô hình đã train trên tập test, chi tiết theo từng lớp (mục IV của báo cáo)
# Chạy sau khi đã train: python scripts/evaluate.py
# Mỗi mô hình lưu vào results/:
#   <tên>_per_class.csv  precision, recall, F1 từng lớp
#   <tên>_pred.csv       nhãn thật và nhãn dự đoán của từng ảnh test, kèm nhãn gốc timeofday, weather
#   <tên>_cm_norm.png    confusion matrix chuẩn hóa theo hàng (tỉ lệ mỗi lớp thật bị đoán thành lớp nào)
#   <tên>_errors.png     một số ảnh bị đoán sai

# 1. Import thư viện
import os
import numpy as np
import pandas as pd
from utils import CLASSES, DATA_DIR, RESULT_DIR, SEED, BDDDataset, evaluate
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import simple_cnn
import complex_cnn
import transfer

BATCH_SIZE = 32
N_ERRORS = 12  # số ảnh đoán sai đưa vào hình

MODELS = [
    ("simple_cnn_noaug", lambda: simple_cnn.build_model(augment=False, name="simple_cnn_noaug")),
    ("simple_cnn", simple_cnn.build_model),
    ("complex_cnn", complex_cnn.build_model),
    ("resnet50", lambda: transfer.build_model()[0]),
]


def save_errors(test_ds, y_true, y_pred, name):
    # lấy ngẫu nhiên N_ERRORS ảnh đoán sai, hiện ảnh 224x224 đúng như mô hình nhìn thấy
    wrong = np.where(y_true != y_pred)[0]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(wrong, size=min(N_ERRORS, len(wrong)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        ax.imshow(np.asarray(Image.open(test_ds.paths[i]).convert("RGB")))
        ax.set_title(f"true: {CLASSES[y_true[i]]}\npred: {CLASSES[y_pred[i]]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, name + "_errors.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    # 2. Load tập test
    test_ds = BDDDataset("test.csv", False, BATCH_SIZE)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    for name, build in MODELS:
        weights = os.path.join(RESULT_DIR, name + "_best.weights.h5")
        if not os.path.exists(weights):
            print("chua co:", weights)
            continue
        print("Danh gia", name)

        # 3. Build model và load checkpoint tốt nhất
        model = build()
        model.load_weights(weights)

        # 4. Dự đoán trên tập test
        y_true, y_pred = evaluate(model, test_ds)
        pred = test_df[["filename", "timeofday", "weather", "label"]].copy()
        pred["pred"] = y_pred
        pred.to_csv(os.path.join(RESULT_DIR, name + "_pred.csv"), index=False)

        # 5. Precision, recall, F1 từng lớp
        rep = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        per_class = pd.DataFrame(rep).T.drop(index="accuracy").round(4)
        per_class["support"] = per_class["support"].astype(int)
        print(per_class)
        per_class.to_csv(os.path.join(RESULT_DIR, name + "_per_class.csv"), index_label="class")

        # 6. Confusion matrix chuẩn hóa theo hàng
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))), normalize="true")
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", xticks_rotation=45,
                                                               values_format=".2f", colorbar=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, name + "_cm_norm.png"))
        plt.close()

        # 7. Ảnh bị đoán sai
        save_errors(test_ds, y_true, y_pred, name)
