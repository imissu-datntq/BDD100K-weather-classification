from ultralytics import YOLO
import cv2

# 1. Tải mô hình YOLOv8 (bản nhỏ, nhanh)
model = YOLO('yolov8n.pt')  # Có thể thay bằng yolov8s.pt, yolov8m.pt,...

# 2. Đọc ảnh đầu vào
image_path = './Data/images_dogs.jpg'  # Đường dẫn ảnh
img = cv2.imread(image_path)

# 3. Dự đoán đối tượng trong ảnh
results = model(image_path)

# 4. Hiển thị kết quả phát hiện
annotated_img = results[0].plot()  # Vẽ khung quanh đối tượng
cv2.imshow('YOLO Detection', annotated_img)

# 5. Nhấn phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()