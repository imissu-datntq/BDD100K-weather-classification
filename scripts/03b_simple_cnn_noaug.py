# Thí nghiệm so sánh: mô hình 1 (CNN đơn giản) nhưng không dùng augmentation
# Giữ nguyên kiến trúc, seed, số epoch, learning rate như 03a_simple_cnn.py, chỉ bỏ lớp augmentation
# Chạy: python scripts/03b_simple_cnn_noaug.py

# 1. Import thư viện
import os
import importlib
import numpy as np
from utils import IMG_SIZE, CLASSES, RESULT_DIR, set_seed, get_loaders, train_model, report
# tên file bắt đầu bằng số nên không viết import thường được
simple_cnn = importlib.import_module("03a_simple_cnn")
build_model, EPOCHS, LR, PATIENCE = simple_cnn.build_model, simple_cnn.EPOCHS, simple_cnn.LR, simple_cnn.PATIENCE

NAME = "simple_cnn_noaug"


if __name__ == "__main__":
    set_seed()

    # 2. Load dữ liệu
    train_ds, val_ds, test_ds = get_loaders()

    # 3. Build model, không có lớp augmentation
    model = build_model(augment=False, name=NAME)
    assert model(np.random.randn(2, IMG_SIZE, IMG_SIZE, 3).astype("float32")).shape == (2, len(CLASSES))
    model.summary()

    # 4. Train, lưu checkpoint có F1 trên val cao nhất
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS, lr=LR, name=NAME, patience=PATIENCE)

    # 5. Evaluate trên tập test bằng checkpoint tốt nhất
    model.load_weights(os.path.join(RESULT_DIR, NAME + "_best.weights.h5"))
    report(model, test_ds, NAME, history)
