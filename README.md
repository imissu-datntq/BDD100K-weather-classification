# Phân loại thời điểm và thời tiết trên BDD100K

Bài tập lớn giữa kỳ môn Deep Learning - Khoa Toán Kinh tế, Đại học Kinh tế Quốc dân.

Thành viên nhóm:
- Nguyễn Trần Quốc Đạt
- Bùi Huỳnh Gia Huy
- Trương Đức Anh
- Ninh Duy Tuân

## Bài toán

Cho một ảnh chụp từ camera hành trình, dự đoán ảnh được chụp vào thời điểm nào trong ngày và trời có quang hay không.

- Input: ảnh RGB chụp từ xe
- Output: 1 trong 6 lớp

| | clear | unclear |
|---|---|---|
| daytime | daytime_clear | daytime_unclear |
| night | night_clear | night_unclear |
| dawn/dusk | dawndusk_clear | dawndusk_unclear |

## Dữ liệu

Dùng bộ BDD100K bản trên Kaggle: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k

Mỗi ảnh trong BDD100K có 2 thuộc tính là `timeofday` và `weather`. Nhóm gộp lại như sau:

- `timeofday`: giữ nguyên `daytime`, `night`, `dawn/dusk`. Bỏ ảnh `undefined`.
- `weather`:
  - clear = `clear`, `partly cloudy`
  - unclear = `overcast`, `rainy`, `snowy`, `foggy`
  - bỏ ảnh `undefined`

Bộ test của BDD100K không có nhãn nên nhóm chia lại như sau:

- Tập val gốc (10k ảnh) dùng làm tập test
- Tập train gốc (70k ảnh) chia 80/20 thành train và validation, `random_state = 42`, có stratify theo nhãn

## Các bước thực hiện

1. Tiền xử lý: làm sạch dữ liệu, resize, chuẩn hóa, tăng cường dữ liệu (augmentation)
2. Chia dữ liệu thành 3 tập train / validation / test
3. Mô hình CNN đơn giản (conv, pooling, fully connected)
4. Mô hình CNN phức tạp hơn, xây từ các block CNN
5. Mô hình dùng transfer learning / fine-tuning
6. Đánh giá trên tập test bằng accuracy, precision, recall, F1-score và confusion matrix

## Cài đặt và tải dữ liệu

Dùng Python 3.12 (3.11 cũng được). Tạo môi trường và cài thư viện:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Code viết bằng Keras 3, chạy trên backend PyTorch để train được bằng GPU trên Windows (TensorFlow bản mới không còn hỗ trợ GPU trên Windows). `requirements.txt` cài torch bản CUDA 12.6. Máy không có GPU NVIDIA vẫn cài được nhưng sẽ train bằng CPU, rất chậm. Cần driver NVIDIA bản 560 trở lên (hỗ trợ CUDA 12.6), kiểm tra bằng `nvidia-smi`.

### Tạo API token Kaggle

Cần có tài khoản Kaggle và `kaggle>=1.8.0` (đã có trong `requirements.txt`).

1. Đăng nhập kaggle.com, vào **Settings → API Tokens** (https://www.kaggle.com/settings/api)
2. Ở mục **API Tokens (Recommended)**, nhập tên token bất kỳ (ví dụ `bdd100k`) rồi bấm **Generate**
3. Copy chuỗi token ngay, vì có thể Kaggle chỉ hiện token một lần

Không bấm phần **Legacy API Credentials** ở dưới, vì nó sẽ huỷ các key `kaggle.json` cũ.

### Lưu token

Token được lưu trong file `~/.kaggle/access_token` (trên Windows là `C:\Users\<tên>\.kaggle\access_token`). Trên Windows chạy 2 lệnh PowerShell sau, thay `<token>` bằng chuỗi vừa copy:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
```

```powershell
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "<token>" -NoNewline -Encoding ascii
```

Lệnh đầu tạo thư mục `.kaggle`, lệnh sau ghi token vào file. Phải có `-Encoding ascii` vì PowerShell 5.1 mặc định ghi UTF-8 có BOM, làm Kaggle đọc sai token.

Trên Linux/macOS:

```bash
mkdir -p ~/.kaggle
echo -n "<token>" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Cách khác là đặt biến môi trường `KAGGLE_API_TOKEN` thay cho file. Trên Windows chạy lệnh sau rồi mở terminal mới:

```powershell
setx KAGGLE_API_TOKEN "<token>"
```

Token là key cá nhân, không commit và không chia sẻ. Mỗi người tự tạo token của mình.

### Tải dữ liệu

Kiểm tra token bằng `kaggle datasets list -s bdd100k`. Nếu in ra danh sách dataset là được. Sau đó tải và giải nén vào `data/raw/` (khoảng vài GB):

```bash
kaggle datasets download solesensei/solesensei_bdd100k -p data/raw --unzip
```

## Cấu trúc thư mục

```
docs/                     yêu cầu đề bài, mẫu báo cáo, code mẫu
scripts/                  đánh số theo thứ tự chạy
  01_prepare_data.py      gộp nhãn 6 lớp, chia train/val/test, resize ảnh về 224x224
  02_explore_data.py      thống kê dữ liệu: số ảnh mỗi lớp, phân bố nhãn gốc, ảnh mẫu
  03a_simple_cnn.py       mô hình 1: CNN đơn giản
  03b_simple_cnn_noaug.py mô hình 1 nhưng không dùng augmentation (thí nghiệm so sánh)
  04_complex_cnn.py       mô hình 2: CNN xây từ các block (conv + BatchNorm + shortcut)
  05_transfer.py          mô hình 3: fine-tune ResNet50 (ImageNet)
  06_evaluate.py          đánh giá chi tiết từng lớp trên tập test
  07_compare.py           bảng so sánh kết quả các mô hình
  utils.py                hàm dùng chung: đọc dữ liệu, augmentation, train, đánh giá, vẽ hình
data/                     dữ liệu (không đưa lên git)
results/                  checkpoint, metrics, hình vẽ (không đưa lên git)
```

## Cách chạy

Chạy từ thư mục gốc của repo, trong môi trường `.venv` đã kích hoạt:

```bash
python scripts/01_prepare_data.py
```

Chạy 1 lần sau khi tải dữ liệu. Script đọc nhãn trong `data/raw/`, ghi `data/train.csv`, `data/val.csv`, `data/test.csv` và ảnh đã resize vào `data/images/` (khoảng 15 phút).

```bash
python scripts/02_explore_data.py
```

Thống kê số ảnh mỗi lớp, phân bố nhãn gốc và ảnh mẫu, lưu vào `results/`.

```bash
python scripts/03a_simple_cnn.py
```

```bash
python scripts/03b_simple_cnn_noaug.py
```

```bash
python scripts/04_complex_cnn.py
```

```bash
python scripts/05_transfer.py
```

Mỗi mô hình lưu checkpoint tốt nhất (theo F1 macro trên val) và kết quả trên tập test vào `results/`: `<tên>_metrics.json`, `<tên>_history.png` (loss, accuracy theo epoch), `<tên>_cm.png` (confusion matrix).

Lần đầu chạy `05_transfer.py` sẽ tự tải trọng số ResNet50 ImageNet (khoảng 100MB) nên cần có mạng.

```bash
python scripts/06_evaluate.py
```

Đánh giá lại từng mô hình đã train trên tập test: precision, recall, F1 từng lớp, confusion matrix chuẩn hóa và ảnh bị đoán sai.

```bash
python scripts/07_compare.py
```

In bảng so sánh 3 mô hình và lưu `results/compare.csv`.

Batch size đang đặt cho GPU: `simple_cnn` (batch 64) và `complex_cnn` (batch 32) chạy được trên GPU 4GB. `transfer` (batch 32) cần GPU khoảng 12GB như RTX 3060. Nếu báo `CUDA out of memory` thì giảm `BATCH_SIZE` ở đầu file.
