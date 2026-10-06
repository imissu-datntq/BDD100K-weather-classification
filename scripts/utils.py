# Các hàm dùng chung cho 3 mô hình: đọc dữ liệu, augmentation, train, đánh giá, vẽ hình
# Các file mô hình phải import utils trước keras để keras chạy bằng backend PyTorch

# 1. Import thư viện
import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc của repo
DATA_DIR = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "results")
IMG_SIZE = 224
BATCH_SIZE = 64
SEED = 42

CLASSES = ["daytime_clear", "daytime_unclear", "night_clear", "night_unclear", "dawndusk_clear", "dawndusk_unclear"]
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")  # chuẩn hóa theo ImageNet
STD = np.array([0.229, 0.224, 0.225], dtype="float32")

# train bằng float16 cho nhanh và đỡ tốn VRAM, lớp cuối của model phải để dtype="float32"
keras.mixed_precision.set_global_policy("mixed_float16")


def set_seed(seed=SEED):
    keras.utils.set_random_seed(seed)  # đặt seed cho random, numpy và torch


# 2. Đọc dữ liệu và augmentation
def get_class_weights():
    # w_c = N / (6 * n_c), lớp ít ảnh thì trọng số lớn
    labels = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))["label"].values
    counts = np.bincount(labels, minlength=len(CLASSES))
    return len(labels) / (len(CLASSES) * counts)


class BDDDataset(keras.utils.PyDataset):
    # mỗi lần lấy 1 batch ảnh đã resize sẵn 224x224, chuẩn hóa theo ImageNet
    # tập train trả thêm trọng số lớp cho từng ảnh và được xáo trộn sau mỗi epoch
    def __init__(self, csv_name, train, batch_size=BATCH_SIZE, workers=4):
        super().__init__(workers=workers, use_multiprocessing=False)
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        self.paths = np.array([os.path.join(DATA_DIR, "images", s, f) for s, f in zip(df["split_dir"], df["filename"])])
        self.labels = df["label"].values
        self.train = train
        self.batch_size = batch_size
        self.weights = get_class_weights()
        self.index = np.arange(len(self.labels))
        self.rng = np.random.default_rng(SEED)
        if train:
            self.rng.shuffle(self.index)

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, i):
        idx = self.index[i * self.batch_size:(i + 1) * self.batch_size]
        x = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype="float32") for p in self.paths[idx]])
        x = (x / 255.0 - MEAN) / STD
        y = self.labels[idx]
        if self.train:
            return x, y, self.weights[y].astype("float32")
        return x, y

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.index)


def get_loaders(batch_size=BATCH_SIZE, workers=4):
    train_ds = BDDDataset("train.csv", True, batch_size, workers)
    val_ds = BDDDataset("val.csv", False, batch_size, workers)
    test_ds = BDDDataset("test.csv", False, batch_size, workers)
    return train_ds, val_ds, test_ds


def get_augmentation():
    # đặt ngay sau Input của mô hình, keras chỉ augment lúc train, lúc đánh giá thì bỏ qua
    # chỉ lật ngang và zoom nhẹ (giữ 80-100% diện tích ảnh)
    # không chỉnh sáng / tương phản / grayscale, không lật dọc vì sẽ làm đổi nhãn ngày đêm, thời tiết
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomZoom(height_factor=(-0.1, 0.0), width_factor=(-0.1, 0.0), seed=SEED),
    ], name="augmentation")


# 3. Train
def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate(model, ds):
    probs = model.predict(ds, verbose=0)
    return ds.labels, probs.argmax(axis=1)


class ValF1(keras.callbacks.Callback):
    # sau mỗi epoch tính loss, acc, F1 macro trên val, để chọn checkpoint theo F1 thay vì accuracy
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.val_ds.labels
        probs = self.model.predict(self.val_ds, verbose=0)
        y_pred = probs.argmax(axis=1)
        loss = -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-7, 1.0))
        logs["val_loss"] = float(np.mean(loss * self.val_ds.weights[y_true]))  # có trọng số lớp giống loss lúc train
        logs["val_acc"] = accuracy_score(y_true, y_pred)
        logs["val_f1"] = macro_f1(y_true, y_pred)
        print(f"epoch {epoch + 1}: train loss {logs['loss']:.4f} acc {logs['accuracy']:.4f} | "
              f"val loss {logs['val_loss']:.4f} acc {logs['val_acc']:.4f} f1 {logs['val_f1']:.4f}")


def train_model(model, train_ds, val_ds, epochs, lr, name, weight_decay=0.0, patience=5):
    os.makedirs(RESULT_DIR, exist_ok=True)
    # chỉ các layer trainable mới được cập nhật, nên phải compile lại sau khi đóng băng / mở khóa
    opt = keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay if weight_decay > 0 else None)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        ValF1(val_ds),
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="max", factor=0.5, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(RESULT_DIR, name + "_best.weights.h5"), monitor="val_f1",
                                        mode="max", save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=patience, verbose=1),
    ]
    start = time.time()
    H = model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    return {
        "train_loss": H.history["loss"],
        "val_loss": H.history["val_loss"],
        "train_acc": H.history["accuracy"],
        "val_acc": H.history["val_acc"],
        "val_f1": H.history["val_f1"],
        "train_time_min": (time.time() - start) / 60,
    }


# 4. Đánh giá trên tập test và vẽ hình
def report(model, test_ds, name, history):
    y_true, y_pred = evaluate(model, test_ds)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": macro_f1(y_true, y_pred),
        "n_params": int(model.count_params()),
        "train_time_min": round(history["train_time_min"], 2),
    }
    print(classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                digits=4, zero_division=0))
    print(metrics)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, name + "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # đường loss và accuracy theo epoch
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, name + "_history.png"))
    plt.close()

    # confusion matrix trên tập test
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, name + "_cm.png"))
    plt.close()
    return metrics


if __name__ == "__main__":
    # kiểm tra nhanh: python scripts/utils.py
    train_ds, val_ds, test_ds = get_loaders()
    x, y, w = train_ds[0]
    assert x.shape == (64, IMG_SIZE, IMG_SIZE, 3)
    assert y.min() >= 0 and y.max() <= 5
    assert len(val_ds[0]) == 2  # val, test không có trọng số lớp
    print("mean / std sau chuan hoa:", x.mean().round(2), x.std().round(2))

    weights = get_class_weights()
    print(weights)
    assert weights.argmax() in (4, 5)  # lớp dawn/dusk phải có trọng số lớn nhất
    assert np.allclose(w, weights[y])

    aug = get_augmentation()
    names = [l.__class__.__name__ for l in aug.layers]
    print(names)
    assert names == ["RandomFlip", "RandomZoom"]  # không chỉnh sáng / tương phản / grayscale
    assert aug.layers[0].mode == "horizontal"  # không lật dọc
    out = keras.ops.convert_to_numpy(aug(x[:4], training=False))
    assert np.abs(out - x[:4]).max() < 1e-2  # lúc đánh giá thì không augment (sai số do float16)

    # F1 macro chỉ tính trên các lớp có mặt: (0.8 + 0) / 2
    assert abs(macro_f1([0, 0, 1], [0, 0, 0]) - 0.4) < 1e-9
    print("ok")
