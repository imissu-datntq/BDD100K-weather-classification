# Chuẩn bị dữ liệu: gộp nhãn thành 6 lớp, chia train/val/test, resize ảnh về 224x224
# Chạy 1 lần: python scripts/01_prepare_data.py
import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc của repo
IMG_ROOT = os.path.join(ROOT, "data/raw/bdd100k/bdd100k/images/100k")
LABEL_DIR = os.path.join(ROOT, "data/raw/bdd100k_labels_release/bdd100k/labels")
OUT_DIR = os.path.join(ROOT, "data")
IMG_SIZE = 224
SEED = 42

CLASSES = ["daytime_clear", "daytime_unclear", "night_clear", "night_unclear", "dawndusk_clear", "dawndusk_unclear"]

TIME_MAP = {"daytime": "daytime", "night": "night", "dawn/dusk": "dawndusk"}
WEATHER_MAP = {
    "clear": "clear", "partly cloudy": "clear",
    "overcast": "unclear", "rainy": "unclear", "snowy": "unclear", "foggy": "unclear",
}


def get_label(timeofday, weather):
    # trả về None nếu ảnh có nhãn undefined (bỏ ảnh)
    if timeofday not in TIME_MAP or weather not in WEATHER_MAP:
        return None
    return CLASSES.index(TIME_MAP[timeofday] + "_" + WEATHER_MAP[weather])


def read_labels(json_path):
    print("Doc", json_path)
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for item in data:
        timeofday = item["attributes"]["timeofday"]
        weather = item["attributes"]["weather"]
        label = get_label(timeofday, weather)
        if label is not None:
            rows.append({"filename": item["name"], "label": label, "timeofday": timeofday, "weather": weather})
    print("  giu lai", len(rows), "/", len(data), "anh")
    del data
    return pd.DataFrame(rows)


def find_images(folder):
    # ảnh train trên Kaggle nằm rải trong nhiều thư mục con (trainA, trainB, ...)
    paths = {}
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(".jpg"):
                paths[name] = os.path.join(root, name)
    return paths


def resize_images(df, src_paths):
    # resize và lưu ảnh, trả về df chỉ gồm những ảnh xử lý được
    ok = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dst = os.path.join(OUT_DIR, "images", row["split_dir"], row["filename"])
        if os.path.exists(dst):
            ok.append(i)
            continue
        src = src_paths.get(row["filename"])
        if src is None:
            continue
        try:
            with Image.open(src) as img:
                img.draft("RGB", (IMG_SIZE, IMG_SIZE))  # giải mã jpeg ở độ phân giải thấp hơn cho nhanh
                img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                img.save(dst, quality=95)
            ok.append(i)
        except Exception as e:
            print("loi anh", src, e)
    return df.loc[ok]


if __name__ == "__main__":
    assert get_label("daytime", "clear") == 0
    assert get_label("daytime", "partly cloudy") == 0
    assert get_label("daytime", "rainy") == 1
    assert get_label("night", "overcast") == 3
    assert get_label("dawn/dusk", "foggy") == 5
    assert get_label("undefined", "clear") is None
    assert get_label("night", "undefined") is None

    # 1. Đọc nhãn
    train_full = read_labels(os.path.join(LABEL_DIR, "bdd100k_labels_images_train.json"))
    test = read_labels(os.path.join(LABEL_DIR, "bdd100k_labels_images_val.json"))

    # 2. Chia tập: val gốc làm test, train gốc chia 80/20
    train, val = train_test_split(train_full, test_size=0.2, random_state=SEED, stratify=train_full["label"])
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split_dir"] = "train"
    val["split_dir"] = "train"
    test["split_dir"] = "test"

    # 3. Resize ảnh, bỏ ảnh thiếu file hoặc lỗi
    os.makedirs(os.path.join(OUT_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "images", "test"), exist_ok=True)
    train_paths = find_images(os.path.join(IMG_ROOT, "train"))
    test_paths = find_images(os.path.join(IMG_ROOT, "val"))

    n_before = len(train) + len(val) + len(test)
    train = resize_images(train, train_paths)
    val = resize_images(val, train_paths)
    test = resize_images(test, test_paths)
    n_skipped = n_before - len(train) - len(val) - len(test)

    cols = ["filename", "split_dir", "label", "timeofday", "weather"]
    train[cols].to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val[cols].to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test[cols].to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    # 4. Đếm số ảnh mỗi lớp
    counts = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).sort_index().fillna(0).astype(int)
    counts.index = [CLASSES[i] for i in counts.index]
    print(counts)
    print("tong:", len(train), len(val), len(test))

    assert len(set(train.filename) & set(val.filename)) == 0
    assert len(set(train.filename) & set(test.filename)) == 0
    assert len(set(val.filename) & set(test.filename)) == 0
    assert abs(len(val) / (len(train) + len(val)) - 0.2) < 0.01
    print("bo qua (thieu file / loi):", n_skipped)
