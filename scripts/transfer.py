# Mô hình 3: transfer learning, fine-tune ResNet50 đã train trên ImageNet
# Chạy: python scripts/transfer.py

# 1. Import thư viện
import os
import numpy as np
from utils import IMG_SIZE, CLASSES, RESULT_DIR, MEAN, STD, set_seed, get_loaders, get_augmentation, train_model, report
from keras import ops
from keras.models import Model
from keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense, Dropout
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

NAME = "resnet50"
BATCH_SIZE = 32
EPOCHS_1, LR_1, PATIENCE_1 = 3, 1e-3, 3  # giai đoạn 1: chỉ train lớp cuối
EPOCHS_2, LR_2, PATIENCE_2 = 10, 1e-4, 3  # giai đoạn 2: fine-tune toàn bộ
WEIGHT_DECAY = 1e-4


def to_caffe(t):
    # dữ liệu đang chuẩn hóa kiểu (x/255 - mean) / std
    # đổi lại về kiểu caffe (thang 0-255, BGR, trừ mean) mà ResNet50 của keras cần
    return preprocess_input((t * ops.convert_to_tensor(STD) + ops.convert_to_tensor(MEAN)) * 255.0)


def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = get_augmentation()(inputs)
    x = Lambda(to_caffe, output_shape=(IMG_SIZE, IMG_SIZE, 3), name="to_caffe", dtype="float32")(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(CLASSES), activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name=NAME), base_model


if __name__ == "__main__":
    set_seed()

    # 2. Load dữ liệu
    train_ds, val_ds, test_ds = get_loaders(batch_size=BATCH_SIZE)

    # 3. Build model
    model, base_model = build_model()
    assert model(np.random.randn(2, IMG_SIZE, IMG_SIZE, 3).astype("float32")).shape == (2, len(CLASSES))

    # 4. Giai đoạn 1: đóng băng ResNet50, chỉ train lớp Dense cuối
    base_model.trainable = False
    model.summary()
    h1 = train_model(model, train_ds, val_ds, epochs=EPOCHS_1, lr=LR_1, name=NAME + "_stage1", patience=PATIENCE_1)

    # 5. Giai đoạn 2: mở khóa toàn bộ, fine-tune với learning rate nhỏ
    base_model.trainable = True
    h2 = train_model(model, train_ds, val_ds, epochs=EPOCHS_2, lr=LR_2, name=NAME,
                     weight_decay=WEIGHT_DECAY, patience=PATIENCE_2)

    # 6. Evaluate trên tập test bằng checkpoint tốt nhất của giai đoạn 2
    model.load_weights(os.path.join(RESULT_DIR, NAME + "_best.weights.h5"))
    history = {k: h1[k] + h2[k] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    history["train_time_min"] = h1["train_time_min"] + h2["train_time_min"]  # tính cả 2 giai đoạn
    report(model, test_ds, NAME, history)
