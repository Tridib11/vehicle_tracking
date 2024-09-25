import time
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# Define the video path
video_path = 'cars.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define target playback FPS
target_fps = 60  # Set to 60 FPS for faster playback
frame_delay = 1.0 / target_fps  # Calculate frame delay

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the list of class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
               'toothbrush']

# Initialize object count dictionary
object_counts = {name: 0 for name in class_names}

# Initialize detected objects tracking
detected_objects = {name: 0 for name in class_names}

# Initialize counters and timer
counter, fps_count, elapsed = 0, 0, 0
start_time = time.perf_counter()

# Set delay for updating the CSV file
delay = 4  # seconds
last_update_time = start_time

# Create or load CSV file
csv_file = 'object_counts.csv'
if not pd.io.common.file_exists(csv_file):
    pd.DataFrame(columns=['Class Name', 'Count']).to_csv(csv_file, index=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video or encountered a read error.")
        break

    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = og_frame.copy()

    # Perform detection with YOLOv8
    results = model(frame, device=device, classes=None, conf=0.25)

    detections = []
    cls = []
    for result in results:
        boxes = result.boxes
        cls = boxes.cls.tolist()
        xywh = boxes.xywh
        conf = boxes.conf.tolist()

        for i, class_index in enumerate(cls):
            x, y, w, h = xywh[i]
            detections.append((x.item(), y.item(), w.item(), h.item(), conf[i]))

    if detections:
        bbox_xywh = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
        confidences = np.array([det[4] for det in detections])

        outputs = tracker.update(bbox_xywh, confidences, frame)

        detected_objects_frame = {name: 0 for name in class_names}
        for output in outputs:
            x1, y1, x2, y2, track_id = output
            class_name = class_names[int(cls[0])] if cls else "Unknown"

            if class_name in detected_objects_frame:
                detected_objects_frame[class_name] += 1

            color = (0, 255, 0)
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(og_frame, f"{class_name} ID: {track_id}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update counts based on detected objects
        for obj, count in detected_objects_frame.items():
            if count > 0:
                object_counts[obj] = max(object_counts[obj], count)

    # Update FPS and place on frame
    current_time = time.perf_counter()
    elapsed = (current_time - start_time)
    counter += 1
    if elapsed > 1:
        fps_count = counter / elapsed
        counter = 0
        start_time = current_time

    # Display FPS on the frame (right side, red color)
    text = f"FPS: {fps_count:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = frame_width - text_width - 10
    text_y = 50
    cv2.putText(og_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display counts of each object on the frame
    y_offset = 30
    for class_name, count in object_counts.items():
        if count > 0:
            cv2.putText(og_frame, f"{class_name}: {count}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 20

    # Check if it's time to update the CSV file
    if current_time - last_update_time >= delay:
        last_update_time = current_time

        # Prepare data for CSV
        df = pd.DataFrame(list(object_counts.items()), columns=['Class Name', 'Count'])
        df.to_csv(csv_file, index=False)

    # Simulate faster playback by adjusting frame delay
    time.sleep(frame_delay)

    cv2.imshow("Live Tracking", og_frame)
    out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
