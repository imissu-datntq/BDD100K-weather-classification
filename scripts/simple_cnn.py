# Mô hình 1: CNN đơn giản (conv, pooling, fully connected), không BatchNorm, không block
# Chạy: python scripts/simple_cnn.py

# 1. Import thư viện
import os
import numpy as np
from utils import IMG_SIZE, CLASSES, RESULT_DIR, set_seed, get_loaders, get_augmentation, train_model, report
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

NAME = "simple_cnn"
EPOCHS = 20
LR = 1e-3
PATIENCE = 5


def build_model():
    model = Sequential(name=NAME)
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(get_augmentation())

    # 4 lần conv 3x3 -> relu -> maxpool, ảnh 224 -> 14x14x128
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # phần fully connected: 14*14*128 = 25088 -> 256 -> 6
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASSES), activation="softmax", dtype="float32"))
    return model


if __name__ == "__main__":
    set_seed()

    # 2. Load dữ liệu
    train_ds, val_ds, test_ds = get_loaders()

    # 3. Build model
    model = build_model()
    assert model(np.random.randn(2, IMG_SIZE, IMG_SIZE, 3).astype("float32")).shape == (2, len(CLASSES))
    model.summary()

    # 4. Train, lưu checkpoint có F1 trên val cao nhất
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS, lr=LR, name=NAME, patience=PATIENCE)

    # 5. Evaluate trên tập test bằng checkpoint tốt nhất
    model.load_weights(os.path.join(RESULT_DIR, NAME + "_best.weights.h5"))
    report(model, test_ds, NAME, history)
