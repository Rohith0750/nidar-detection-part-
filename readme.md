# 📌 Human Detection with YOLOv8 (GUI-Based)
This project is a Python-based desktop application that detects humans in a video file using Ultralytics YOLOv8 and OpenCV, with a simple GUI built using Tkinter.

It supports tracking, bounding box visualization, and saves the processed video with detected humans.

## 🔍 Features
🖼️ Graphical User Interface (GUI) for easy video selection

✅ Human detection using YOLOv8

🧠 Real-time object tracking with confidence scores

💾 Output saved as annotated video

📌 Only detects class 0 (person)

## 🛠️ Requirements
Ensure you have Python 3.8 or higher installed.

Install the required packages using:
    pip install ultralytics opencv-python tkinter

⚠️ tkinter is built-in with most Python installations. If not, install via your OS package manager.

# 🧠 How It Works
The GUI allows users to select any .mp4 video file.

When you click "Run Detection", the app:

Loads the YOLOv8 model (yolo11x.pt)

Resizes frames to 640×480

Detects only humans (class 0) with a confidence threshold (default: 40%)

Annotates detected persons with bounding boxes and track IDs

## The resulting annotated video is saved to:
 outputDetected/output_detected.mp4
You get a popup message once processing completes — whether humans were detected or not.

## ▶️ How to Use
Run the script:


python main.py

Click "Browse Video" to select a .mp4 file.

Click "Run Detection" to start processing.

A window will show real-time frame processing. Press q to stop early.

## 📂 Output
The output video with bounding boxes is saved to:

outputDetected/output_detected.mp4

Frames are resized to 640x480 for consistency and processing speed.

## ⚙️ Configurations
You can tweak:

CONFIDENCE_THRESHOLD to adjust detection sensitivity

model = YOLO("yolo11x.pt") to use other YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.

classes=[0] if you want to detect more than just people (e.g., cars, dogs, etc.)

🧪 Tested On
Python 3.10

Windows 11

YOLO11x (ultralytics)

OpenCV 4.x

## 🧱 Next Work

1. Integrate LRF (Laser Range Finder)
Use LRF data to measure the actual distance to each detected survivor.

Combine this with bounding box location and camera intrinsics to triangulate GPS coordinates.

2. Compute Real-World GPS from Bounding Boxes
Using drone’s current GPS + camera orientation + LRF distance + bounding box:

Compute latitude and longitude of each survivor.

This is essentially georeferencing detections.

3. Output Survivor GPS List
Maintain a list of detected survivor GPS locations.

This list is consumed by Pranav’s module for:

Deciding where to drop payloads.

Planning reflight locations.
