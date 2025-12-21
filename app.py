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

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

# ================== VIDEO URLS ==================
video_urls = [
    "http://localhost:3000/hls/output1.m3u8",
    "http://localhost:3000/hls/output2.m3u8",
    "http://localhost:3000/hls/output3.m3u8",
    "http://localhost:3000/hls/output4.m3u8"
]

caps = [VideoCaptureAsync(url) for url in video_urls]

# ================== CONFIG ==================
small_width, small_height = 426, 240
frame_skip = 3
count = 0
scale_box = 0.84

GREEN_TIME, YELLOW_TIME, RED_TIME = 90, 5, 60
TOTAL_TIME = GREEN_TIME + YELLOW_TIME + RED_TIME
start_time = time.time()

stop_line_thickness = 2
stop_area_points = np.array([
    [int(200 * small_width / 960), int(210 * small_height / 520)],
    [int(590 * small_width / 960), int(320 * small_height / 520)],
    [int(572 * small_width / 960), int(370 * small_height / 520)],
    [int(165 * small_width / 960), int(245 * small_height / 520)]
], np.int32).reshape((-1, 1, 2))

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
        while True:
            if self.frame is not None:
                with self.lock:
                    try:
                        self.results = model(
                            self.frame,
                            verbose=False,
                            device="CPU"
                        )
                    except Exception:
                        self.results = None
            time.sleep(0.001)

yolo_threads = [YOLOThread() for _ in video_urls]

# ================== GENERATE VIDEO FRAMES ==================
def generate_frames(index):
    global count
    cap = caps[index]
    yolo_thread = yolo_threads[index]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        count += 1
        if count % frame_skip != 0:
            continue

        frame_small = cv2.resize(
            frame,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA
        )
        
        yolo_thread.frame = frame_small

        with yolo_thread.lock:
            results = yolo_thread.results

        if results is None:
            continue

        motors, helmets = [], []

        elapsed = int((time.time() - start_time) % TOTAL_TIME)
        if elapsed < GREEN_TIME:
            light_color, light_text = (0, 255, 0), "Hijau - GO"
        elif elapsed < GREEN_TIME + YELLOW_TIME:
            light_color, light_text = (0, 255, 255), "Kuning - SIAP"
        else:
            light_color, light_text = (0, 0, 255), "Merah - BERHENTI"

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label_name = class_list[int(cls)]

                if label_name == "helmet":
                    helmets.append((x1, y1, x2, y2))
                else:
                    motors.append((x1, y1, x2, y2))

                color = (0, 255, 0) if label_name == "helmet" else (0, 0, 255)
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(
                    frame_small,
                    f'{label_name} {conf:.2f}',
                    (x1, y1),
                    scale=0.5,
                    thickness=1,
                    colorR=color
                )

        # ================= STOP LINE HANYA UNTUK VIDEO PERTAMA ==================
        if index == 0:
            cv2.polylines(frame_small, [stop_area_points], True, light_color, stop_line_thickness)
            cv2.putText(
                frame_small,
                light_text,
                (stop_area_points[0][0][0], stop_area_points[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                light_color,
                2
            )

        ret, buffer = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================== FLASK ROUTES ==================
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/video1')
def video1_feed():
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2_feed():
    return Response(generate_frames(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video3')
def video3_feed():
    return Response(generate_frames(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video4')
def video4_feed():
    return Response(generate_frames(3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, threaded=True)