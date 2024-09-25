from ultralytics import YOLO

# Load and automatically download YOLOv8n (nano version) model weights
model = YOLO("yolov8n.pt")  # This will download the model if not available locally

