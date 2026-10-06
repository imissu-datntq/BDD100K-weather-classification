# Mô hình 2: CNN phức tạp hơn, xây từ các block CNN (2 conv + BatchNorm + shortcut)
# Chạy: python scripts/complex_cnn.py

# 1. Import thư viện
import os
import numpy as np
from utils import IMG_SIZE, CLASSES, RESULT_DIR, set_seed, get_loaders, get_augmentation, train_model, report
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

NAME = "complex_cnn"
BATCH_SIZE = 32  # train trên RTX 3060 12GB (~10GB VRAM), batch 64 bị tràn VRAM vì 2 block đầu chạy ở độ phân giải 224 và 112
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5


def conv_block(x, out_ch, name):
    # nhánh chính: conv 3x3 -> BN -> ReLU -> conv 3x3 -> BN
    y = Conv2D(out_ch, (3, 3), padding="same")(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(out_ch, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)

    # shortcut: conv 1x1 -> BN để khớp số kênh
    s = Conv2D(out_ch, (1, 1))(x)
    s = BatchNormalization()(s)

    out = Activation("relu")(Add()([y, s]))
    return MaxPooling2D(pool_size=(2, 2), name=name)(out)


def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = get_augmentation()(inputs)

    # stem: conv 3x3 (3 -> 32)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 4 block: 32 -> 64 -> 128 -> 256 -> 512, ảnh 224 -> 14x14x512
    for i, out_ch in enumerate([64, 128, 256, 512]):
        x = conv_block(x, out_ch, name=f"block{i + 1}_out")

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(CLASSES), activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name=NAME)


if __name__ == "__main__":
    set_seed()

    # 2. Load dữ liệu
    train_ds, val_ds, test_ds = get_loaders(batch_size=BATCH_SIZE)

    # 3. Build model
    model = build_model()
    assert model(np.random.randn(2, IMG_SIZE, IMG_SIZE, 3).astype("float32")).shape == (2, len(CLASSES))
    assert model.get_layer("block4_out").output.shape[1:] == (14, 14, 512)
    model.summary()

    # 4. Train, lưu checkpoint có F1 trên val cao nhất
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS, lr=LR, name=NAME,
                          weight_decay=WEIGHT_DECAY, patience=PATIENCE)

    # 5. Evaluate trên tập test bằng checkpoint tốt nhất
    model.load_weights(os.path.join(RESULT_DIR, NAME + "_best.weights.h5"))
    report(model, test_ds, NAME, history)
