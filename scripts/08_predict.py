# Dự đoán bằng các mô hình đã train và ghi ra file models/<tên>.keras, không cần train lại
# (giống Predict.py trong code mẫu, nhưng Keras ghi model bằng model.save thay cho pickle)
# Chạy sau 06_evaluate.py (bước ghi model ra file):
#   python scripts/08_predict.py                 dự đoán tập test, in accuracy, precision, recall, F1 từng mô hình
#   python scripts/08_predict.py a.jpg b.jpg     dự đoán thêm lớp của các ảnh bất kỳ

# 1. Import thư viện
import os
import sys
import importlib
import numpy as np
import pandas as pd
from utils import CLASSES, IMG_SIZE, MEAN, STD, MODEL_DIR, BDDDataset, evaluate
import keras
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# ResNet50 có lớp Lambda gọi hàm to_caffe, phải import file này để keras tìm được hàm khi load model
importlib.import_module("05_transfer")

BATCH_SIZE = 32
NAMES = ["simple_cnn_noaug", "simple_cnn", "complex_cnn", "resnet50"]


def load_images(paths):
    # resize về 224x224 như 01_prepare_data.py, chuẩn hóa như BDDDataset
    x = []
    for p in paths:
        with Image.open(p) as img:
            img.draft("RGB", (IMG_SIZE, IMG_SIZE))
            x.append(np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32"))
    return (np.stack(x) / 255.0 - MEAN) / STD


if __name__ == "__main__":
    # 2. Đọc dữ liệu: tập test và các ảnh truyền vào từ dòng lệnh (nếu có)
    test_ds = BDDDataset("test.csv", False, BATCH_SIZE)
    image_paths = sys.argv[1:]
    x_images = load_images(image_paths) if image_paths else None

    rows = []
    for name in NAMES:
        filename = os.path.join(MODEL_DIR, name + ".keras")
        if not os.path.exists(filename):
            print("chua co:", filename)
            continue

        # 3. Load model từ file, chỉ dự đoán nên không cần optimizer (compile=False)
        model = keras.models.load_model(filename, compile=False)

        # 4. Dự báo nhãn của tập test và đánh giá
        y_true, y_pred = evaluate(model, test_ds)
        print("=" * 25, name, "=" * 25)
        print(classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                    digits=4, zero_division=0))
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        })

        # 5. Dự báo lớp của các ảnh truyền vào
        if x_images is not None:
            probs = model.predict(x_images, verbose=0)
            for p, prob in zip(image_paths, probs):
                print(f"{os.path.basename(p)}: {CLASSES[prob.argmax()]} ({prob.max():.2%})")
            print()

    # 6. Bảng tổng hợp kết quả trên tập test
    if rows:
        print(f"Ket qua tren tap test ({len(test_ds.labels)} anh):")
        print(pd.DataFrame(rows).set_index("model").round(4).to_string())
