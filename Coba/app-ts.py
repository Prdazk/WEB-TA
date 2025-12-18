import cv2
from ultralytics import YOLO
from threading import Thread
import os
import subprocess
import time
from dotenv import load_dotenv

# ================= ENV =================
load_dotenv()
db_path = os.getenv("MODELS_PATH")  # path model YOLO

# ================= CONFIG =================
VIDEO_URL = "http://localhost:3000/hls/cctv_villabs_id_streamer_jsmpeg_streamer_ikip2/output.m3u8"
STREAM_NAME = "helmet_pnm"
OUTPUT_DIR = f"./output/{STREAM_NAME}"

FPS = 15
WIDTH, HEIGHT = 960, 520
FRAME_SKIP = 3  # skip frame agar CPU tidak overload

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= VIDEO ASYNC =================
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
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# ================= IOU (OPTIONAL) =================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ================= YOLO MODEL =================
model = YOLO(db_path)

# ================= COLOR MAP =================
CLASS_COLORS = {
    "helmet": (0, 255, 0),        # hijau
    "no_helmet": (0, 0, 255),     # merah
}

def get_color(label):
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = abs(hash(label)) % 255
    return ((h * 3) % 255, (h * 7) % 255, (h * 11) % 255)

# ================= VIDEO SOURCE =================
cap = VideoCaptureAsync(VIDEO_URL).start()

# ================= FFMPEG → HLS =================
ffmpeg = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", str(FPS),
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "10",
        "-hls_flags", "delete_segments+independent_segments",
        f"{OUTPUT_DIR}/output.m3u8"
    ],
    stdin=subprocess.PIPE
)

# ================= MAIN LOOP =================
last_time = 0
count = 0

print("🚀 Helmet detection started...")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        count += 1
        if count % FRAME_SKIP != 0:
            continue

        now = time.time()
        if now - last_time < 1 / FPS:
            continue
        last_time = now

        frame_small = cv2.resize(frame, (WIDTH, HEIGHT))

        results = model(frame_small, verbose=False, device="cpu")  # gunakan CPU atau "cuda" jika GPU tersedia

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls]

                color = get_color(label)
                thickness = 3 if label == "no_helmet" else 2

                cv2.rectangle(frame_small, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame_small, f"{label.upper()} {conf:.2f}",
                            (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # ==== SEND TO HLS ====
        if frame_small is not None and frame_small.size > 0:
            try:
                ffmpeg.stdin.write(frame_small.tobytes())
            except BrokenPipeError:
                time.sleep(0.01)
                continue

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    cap.stop()
    if ffmpeg.stdin:
        ffmpeg.stdin.close()
    ffmpeg.wait()
    print("✅ Clean exit")
