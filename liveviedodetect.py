import cv2
import os
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11x.pt")  # Make sure this file exists

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.4

# Set webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

# Output video path setup
output_dir = "output_detected"
os.makedirs(output_dir, exist_ok=True)

# Unique filename using timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"detected_{timestamp}.mp4"
output_path = os.path.join(output_dir, output_filename)

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"🎥 Saving output to: {output_path}")

# Start detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLOv8 tracking
    results = model.track(
        source=frame_resized,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        classes=[0],  # Human class
        verbose=False
    )

    # Draw results
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {track_id} | {conf:.2f}"
                cv2.putText(frame_resized, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show and save
    cv2.imshow("Live Human Detection", frame_resized)
    out.write(frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection completed and video saved.")
