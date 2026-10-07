# Gom kết quả của 3 mô hình thành bảng so sánh (mục IV của báo cáo)
# Chạy sau khi đã train xong: python scripts/compare.py
import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt khi ghi output ra file trên Windows

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS = [
    ("simple_cnn", "CNN đơn giản"),
    ("complex_cnn", "CNN phức tạp"),
    ("resnet50", "ResNet50 (transfer learning)"),
]

rows = []
for name, label in MODELS:
    path = os.path.join(RESULT_DIR, name + "_metrics.json")
    if not os.path.exists(path):
        print("chua co ket qua:", path)
        continue
    with open(path) as f:
        m = json.load(f)
    rows.append({
        "Model": label,
        "Accuracy": f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1-score": f"{m['f1']:.4f}",
        "Số tham số": f"{m['n_params']:,}",
        "Thời gian train (phút)": round(m["train_time_min"], 1),
    })

if not rows:
    print("chua co ket qua nao, hay train cac mo hinh truoc")
    raise SystemExit

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULT_DIR, "compare.csv"), index=False, encoding="utf-8-sig")

# in thêm dạng bảng markdown để dán vào README
print()
print("| " + " | ".join(df.columns) + " |")
print("|" + "---|" * len(df.columns))
for _, r in df.iterrows():
    print("| " + " | ".join(str(v) for v in r.values) + " |")
