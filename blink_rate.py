import cv2
import time
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO

# -----------------------------
# 1. Load YOLOv8n (auto-download)
# -----------------------------
face_model = YOLO("yolov8n.pt")

# -----------------------------
# 2. Pretrained ResNet18 Eye-State Classifier
# -----------------------------
class EyeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(512, 2)  # open / closed

    def forward(self, x):
        return self.base(x)

eye_model = EyeClassifier()
eye_model.eval()

# -----------------------------
# 3. Preprocessing
# -----------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Blink Logic
# -----------------------------
BLINK_THRESHOLD = 2
FRAME_SKIP = 2

VIDEO_FOLDER = "videos"

total_blinks_all = 0
total_time_all = 0
video_count = 0

for file in os.listdir(VIDEO_FOLDER):

    if not file.lower().endswith(".mp4"):
        continue

    path = os.path.join(VIDEO_FOLDER, file)
    cap = cv2.VideoCapture(path)

    print(f"\nProcessing: {file}")

    blink_count = 0
    closed_frames = 0
    eyes_were_closed = False

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        # -----------------------------
        # YOLO face detection
        # -----------------------------
        results = face_model(frame, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        # Take the first detected face
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy().astype(int)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # -----------------------------
        # Extract eye region (upper 40% of face)
        # -----------------------------
        h = face.shape[0]
        eye_region = face[0:int(h * 0.4), :]

        if eye_region.size == 0:
            continue

        # -----------------------------
        # Eye-state classification
        # -----------------------------
        img = transform(eye_region).unsqueeze(0)
        with torch.no_grad():
            pred = eye_model(img)
            cls = pred.argmax().item()

        eyes_closed = (cls == 1)

        # -----------------------------
        # Blink logic
        # -----------------------------
        if eyes_closed:
            closed_frames += 1
            eyes_were_closed = True
        else:
            if eyes_were_closed and closed_frames >= BLINK_THRESHOLD:
                blink_count += 1
            closed_frames = 0
            eyes_were_closed = False

    elapsed = time.time() - start_time
    rate = blink_count / elapsed if elapsed > 0 else 0

    print(f"  Blinks: {blink_count}")
    print(f"  Time: {elapsed:.2f} sec")
    print(f"  Rate: {rate:.4f} blinks/sec")

    total_blinks_all += blink_count
    total_time_all += elapsed
    video_count += 1

cap.release()

if video_count > 0:
    print("\n===== FINAL OVERALL RESULT =====")
    print(f"Videos Processed : {video_count}")
    print(f"Total Blinks     : {total_blinks_all}")
    print(f"Total Time       : {total_time_all:.2f} sec")
    print(f"Average Rate     : {total_blinks_all / total_time_all:.4f} blinks/sec")
else:
    print("No videos found.")