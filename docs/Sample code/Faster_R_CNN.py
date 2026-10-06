import torch
import torchvision
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load pretrained Faster R-CNN
# -------------------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# -------------------------------
# 2️⃣ Load image
# -------------------------------
image_path = "images_dogs.jpg"  # ảnh có sẵn trong thư mục
image = Image.open(image_path).convert("RGB")

# -------------------------------
# 3️⃣ Transform image to tensor
# -------------------------------
transform = torchvision.transforms.ToTensor()
img_tensor = transform(image)

# -------------------------------
# 4️⃣ Perform object detection
# -------------------------------
with torch.no_grad():
    predictions = model([img_tensor])

# -------------------------------
# 5️⃣ Visualize results
# -------------------------------
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Giữ lại các box có độ tin cậy > 0.7
threshold = 0.7
keep = scores > threshold

draw = ImageDraw.Draw(image)

for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 10), f"{label.item()} ({score:.2f})", fill="red")

# -------------------------------
# 6️⃣ Show image
# -------------------------------
plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.show()
