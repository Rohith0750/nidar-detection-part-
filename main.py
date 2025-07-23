import os
import cv2
from ultralytics import YOLO
from tkinter import filedialog, Tk, Label, Button, messagebox
import threading

# Load YOLOv8 model
model = YOLO("yolo11x.pt")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.4

# GUI Setup
class HumanDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection with YOLO11x")
        self.video_path = None

        self.label = Label(root, text="Select a video file to start detection")
        self.label.pack(pady=10)

        self.select_button = Button(root, text="Browse Video", command=self.browse_video)
        self.select_button.pack(pady=5)

        self.run_button = Button(root, text="Run Detection", command=self.start_detection, state='disabled')
        self.run_button.pack(pady=5)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            self.label.config(text=f"Selected: {os.path.basename(path)}")
            self.run_button.config(state='normal')

    def start_detection(self):
        self.run_button.config(state='disabled')
        thread = threading.Thread(target=self.run_detection)
        thread.start()

    def run_detection(self):
        input_path = self.video_path
        if not os.path.exists(input_path):
            messagebox.showerror("Error", "File not found.")
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Error opening video file.")
            return

        frame_width = 640
        frame_height = 480
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = "outputDetected\output_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        human_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (frame_width, frame_height))
            results = model.predict(
                frame_resized,
                conf=CONFIDENCE_THRESHOLD,
                classes=[0],
                verbose=False
            )

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                        human_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person:{track_id} Conf:{conf:.2f}"
                        cv2.putText(frame_resized, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out.write(frame_resized)
            cv2.imshow("Human Detection", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if human_detected:
            messagebox.showinfo("Done", "✅ Human(s) detected. Saved as output_detected.mp4.")
        else:
            messagebox.showinfo("Done", "❌ No humans detected in the video.")

        self.run_button.config(state='normal')


# Run GUI
if __name__ == "__main__":
    root = Tk()
    app = HumanDetectionApp(root)
    root.geometry("400x200")
    root.mainloop()
