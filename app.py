import cv2
import pandas as pd
from ultralytics import YOLO
from threading import Thread
import os
from dotenv import load_dotenv
import cvzone
import numpy as np
import time

load_dotenv()
db_path = os.getenv('MODELS_PATH')

class VideoCaptureAsync:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# Load YOLO model
model = YOLO(db_path)

# Load class list
try:
    with open("./coco.txt", "r") as f:
        class_list = f.read().strip().split("\n")
except FileNotFoundError:
    class_list = ["person", "bicycle", "car", "motorcycle", "helmet"]

video_url = "http://localhost:3000/hls/output.m3u8"
cap = VideoCaptureAsync(video_url).start()

small_width, small_height = 960, 520
frame_skip = 3
count = 0
scale_box = 0.87

# ================== TRAFFIC LIGHT TIMER ==================
GREEN_TIME = 90
YELLOW_TIME = 5
RED_TIME = 60
TOTAL_TIME = GREEN_TIME + YELLOW_TIME + RED_TIME

start_time = time.time()
# =========================================================

# STOP AREA (TIDAK DIUBAH)
stop_line_thickness = 3
stop_area_points = np.array([
    [200, 210],
    [590, 320],
    [572, 370],
    [165, 245]
], np.int32).reshape((-1, 1, 2))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    count += 1
    if count % frame_skip != 0:
        continue

    frame_small = cv2.resize(frame, (small_width, small_height))
    results = model(frame_small, verbose=False, device="CPU")

    motors = []
    helmets = []

    # ================== TRAFFIC LIGHT STATE ==================
    elapsed = int((time.time() - start_time) % TOTAL_TIME)

    if elapsed < GREEN_TIME:
        stop_line_color = (0, 255, 0)
        light_text = "GREEN - GO"
    elif elapsed < GREEN_TIME + YELLOW_TIME:
        stop_line_color = (0, 255, 255)
        light_text = "YELLOW - READY"
    else:
        stop_line_color = (0, 0, 255)
        light_text = "RED - STOP"
    # =========================================================

    for r in results:
        boxes_data = pd.DataFrame(r.boxes.data).astype(float)
        for _, row in boxes_data.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            conf = float(row[4])
            cls = int(row[5])

            label_name = class_list[cls] if cls < len(class_list) else str(cls)

            if label_name == "helmet":
                helmets.append((x1, y1, x2, y2))
            else:
                motors.append((x1, y1, x2, y2))

            w = x2 - x1
            h = y2 - y1
            x1_adj = int(x1 + (1 - scale_box) / 2 * w)
            y1_adj = int(y1 + (1 - scale_box) / 2 * h)
            x2_adj = int(x2 - (1 - scale_box) / 2 * w)
            y2_adj = int(y2 - (1 - scale_box) / 2 * h)

            color = (0, 255, 0) if label_name == "helmet" else (0, 0, 255)
            cv2.rectangle(frame_small, (x1_adj, y1_adj), (x2_adj, y2_adj), color, 2)
            cvzone.putTextRect(
                frame_small,
                f'{label_name} {conf:.2f}',
                (x1_adj, y1_adj),
                scale=0.8,
                thickness=1,
                colorR=color
            )

    for mx1, my1, mx2, my2 in motors:
        has_helmet = any(
            iou((mx1, my1, mx2, my2), h) > 0.10 for h in helmets
        )

        w = mx2 - mx1
        h = my2 - my1
        mx1_adj = int(mx1 + (1 - scale_box) / 2 * w)
        my1_adj = int(my1 + (1 - scale_box) / 2 * h)
        mx2_adj = int(mx2 - (1 - scale_box) / 2 * w)
        my2_adj = int(my2 - (1 - scale_box) / 2 * h)

        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        label_text = "Helmet" if has_helmet else "No Helmet"
        cv2.rectangle(frame_small, (mx1_adj, my1_adj), (mx2_adj, my2_adj), color, 2)
        cv2.putText(
            frame_small,
            label_text,
            (mx1_adj, my1_adj - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # ================== DRAW STOP AREA ==================
    cv2.polylines(
        frame_small,
        [stop_area_points],
        True,
        stop_line_color,
        stop_line_thickness
    )

    cv2.putText(
        frame_small,
        light_text,
        (stop_area_points[0][0][0], stop_area_points[0][0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        stop_line_color,
        2
    )

    cv2.imshow("Helmet Detection", frame_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.stop()
cv2.destroyAllWindows()
