# mini_project_cv
**Overview-blink rate**

This project performs blink detection on videos using:
- YOLOv8n for fast face detection (auto‑downloaded)
- ResNet18 (ImageNet‑pretrained) as a lightweight eye‑state classifier
- A simple blink‑counting algorithm based on consecutive closed‑eye frames
The script processes all .mp4 files inside the videos/ folder and prints:
- Blink count
- Processing time
- Blink rate (blinks per second)
- Overall totals across all videos

**Dependencies**

pip install ultralytics torch torchvision opencv-python

**Overview-dimensions measurement **

This script performs basic facial measurements from a single image (face.png) using:
- OpenCV Haar Cascade for face detection
- A fallback center‑crop detector if Haar fails
- Simple geometric rules to estimate:
- Face width & height
- Eye width
- Eye distance
- Nose width & height
- Mouth width
It then converts pixel measurements into centimeters using an assumed real‑world face width (default: 15 cm).
This is useful for quick approximations, demos, or assignments involving pixel‑to‑cm conversion.

**Dependencies**

pip install opencv-python numpy
