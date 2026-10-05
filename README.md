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

## Cấu trúc thư mục

```
docs/        yêu cầu đề bài và mẫu báo cáo
```

(sẽ cập nhật thêm khi có code)
