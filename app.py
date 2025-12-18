import cv2
from ultralytics import YOLO
from threading import Thread, Lock
import os
from dotenv import load_dotenv
import cvzone
import numpy as np
import time
from flask import Flask, Response, send_from_directory

load_dotenv()
db_path = os.getenv('MODELS_PATH')

# ================== VIDEO CAPTURE ==================
class VideoCaptureAsync:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video stream: {src}")
        self.ret, self.frame = self._safe_read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def _safe_read(self):
        try:
            ret, frame = self.cap.read()
        except cv2.error:
            ret, frame = False, None
        return ret, frame

    def update(self):
        while self.running:
            self.ret, self.frame = self._safe_read()
            time.sleep(0.001)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# ================== IOU ==================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# ================== LOAD MODEL ==================
model = YOLO(db_path)

try:
    with open("./coco.txt", "r") as f:
        class_list = f.read().strip().split("\n")
except FileNotFoundError:
    class_list = ["person", "bicycle", "car", "motorcycle", "helmet"]

video_url = "http://localhost:3000/hls/output.m3u8"
cap = VideoCaptureAsync(video_url)

# ================== CONFIG ==================
small_width, small_height = 960, 520
frame_skip = 3
count = 0
scale_box = 0.85

GREEN_TIME, YELLOW_TIME, RED_TIME = 90, 5, 60
TOTAL_TIME = GREEN_TIME + YELLOW_TIME + RED_TIME
start_time = time.time()

stop_line_thickness = 3
stop_area_points = np.array([[200,210],[590,320],[572,370],[165,245]], np.int32).reshape((-1,1,2))

# ================== FLASK APP ==================
app = Flask(__name__, static_folder='public', static_url_path='')

# ================== THREADING YOLO ==================
class YOLOThread:
    def __init__(self):
        self.results = None
        self.frame = None
        self.lock = Lock()
        Thread(target=self.run, daemon=True).start()

    def run(self):
        global cap
        while True:
            if self.frame is not None:
                with self.lock:
                    try:
                        self.results = model(self.frame, verbose=False, device="CPU")
                    except Exception:
                        self.results = None
            time.sleep(0.01)

yolo_thread = YOLOThread()

# ================== GENERATE VIDEO FRAMES ==================
def generate_frames():
    global count, yolo_thread
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        count += 1
        if count % frame_skip != 0:
            continue

        frame_small = cv2.resize(frame, (small_width, small_height))
        yolo_thread.frame = frame_small.copy()

        with yolo_thread.lock:
            results = yolo_thread.results
        if results is None:
            time.sleep(0.01)
            continue

        motors, helmets = [], []

        elapsed = int((time.time() - start_time) % TOTAL_TIME)
        if elapsed < GREEN_TIME:
            stop_line_color, light_text = (0,255,0), "GREEN - GO"
        elif elapsed < GREEN_TIME + YELLOW_TIME:
            stop_line_color, light_text = (0,255,255), "YELLOW - READY"
        else:
            stop_line_color, light_text = (0,0,255), "RED - STOP"

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls = int(cls)
                label_name = class_list[cls] if cls < len(class_list) else str(cls)

                if label_name == "helmet":
                    helmets.append((x1, y1, x2, y2))
                else:
                    motors.append((x1, y1, x2, y2))

                w, h = x2 - x1, y2 - y1
                x1_adj = int(x1 + (1-scale_box)/2 * w)
                y1_adj = int(y1 + (1-scale_box)/2 * h)
                x2_adj = int(x2 - (1-scale_box)/2 * w)
                y2_adj = int(y2 - (1-scale_box)/2 * h)

                color = (0,255,0) if label_name=="helmet" else (0,0,255)
                cv2.rectangle(frame_small, (x1_adj, y1_adj), (x2_adj, y2_adj), color, 2)
                cvzone.putTextRect(frame_small, f'{label_name} {conf:.2f}', (x1_adj, y1_adj),
                                   scale=0.6, thickness=1, colorR=color)

        for mx1,my1,mx2,my2 in motors:
            has_helmet = any(iou((mx1,my1,mx2,my2), h)>0.1 for h in helmets)
            w, h = mx2 - mx1, my2 - my1
            mx1_adj = int(mx1 + (1-scale_box)/2 * w)
            my1_adj = int(my1 + (1-scale_box)/2 * h)
            mx2_adj = int(mx2 - (1-scale_box)/2 * w)
            my2_adj = int(my2 - (1-scale_box)/2 * h)
            color = (0,255,0) if has_helmet else (0,0,255)
            label_text = "Helmet" if has_helmet else "No Helmet"
            cv2.rectangle(frame_small, (mx1_adj,my1_adj), (mx2_adj,my2_adj), color, 2)
            cv2.putText(frame_small, label_text, (mx1_adj,my1_adj-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.polylines(frame_small, [stop_area_points], True, stop_line_color, stop_line_thickness)
        cv2.putText(frame_small, light_text, (stop_area_points[0][0][0], stop_area_points[0][0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, stop_line_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame_small)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================== FLASK ROUTES ==================
@app.route('/')
def index():
    # Kirimkan file index.html dari folder public
    return send_from_directory('public', 'index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, threaded=True)
