import os
import threading
import time

# Prevent Linux/Windows backend video enumeration hangs & memory leaks
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
DEFAULT_VIDEO = os.path.join(BASE_DIR, 'sample_video.mp4')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Start in idle state ('none') so Render does not crash from auto-running video on boot
current_mode = 'none'
active_video_path = None
latest_detections = []


class ThreadedCamera:
    """Thread-safe camera capture for local or container environments"""
    def __init__(self):
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Notice: No physical webcam detected on server hardware.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stopped = False

        threading.Thread(target=self.update, daemon=True).start()
        time.sleep(0.5)
        return True

    def update(self):
        while not self.stopped and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.frame = None


camera_thread = ThreadedCamera()


def calculate_severity(box_area, frame_area):
    ratio = (box_area / frame_area) * 100
    if ratio < 1.5:
        return 'Low', (0, 255, 0)       # Green
    elif ratio < 5.0:
        return 'Medium', (255, 165, 0)  # Cyan/Blue
    else:
        return 'High', (0, 0, 255)      # Red


def draw_custom_boxes(frame, results):
    global latest_detections
    h, w = frame.shape[:2]
    frame_area = h * w
    current_frame_detections = []

    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = model.names.get(cls_id, 'Pothole')

            box_area = (x2 - x1) * (y2 - y1)
            severity, color = calculate_severity(box_area, frame_area)

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw Label Tag
            tag = f"{label_name} | {conf:.2f} | {severity}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            current_frame_detections.append({
                'type': label_name,
                'confidence': f"{conf:.2f}",
                'severity': severity
            })

    if current_frame_detections:
        latest_detections = current_frame_detections

    return frame, current_frame_detections


def generate_frames():
    global current_mode, active_video_path

    if current_mode == 'webcam':
        success = camera_thread.start()
        if not success:
            return

        while current_mode == 'webcam':
            frame = camera_thread.read()
            if frame is None:
                time.sleep(0.03)
                continue

            with torch.no_grad():
                results = model(frame, conf=0.25, imgsz=480, verbose=False)

            annotated_frame, _ = draw_custom_boxes(frame, results)
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        camera_thread.stop()

    elif current_mode == 'video' and active_video_path and os.path.exists(active_video_path):
        cap = cv2.VideoCapture(active_video_path)
        while current_mode == 'video' and cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            with torch.no_grad():
                results = model(frame, conf=0.25, imgsz=480, verbose=False)

            annotated_frame, _ = draw_custom_boxes(frame, results)
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()


@app.route('/')
def index():
    return render_template('index.html', mode=current_mode)


@app.route('/start_webcam')
def start_webcam():
    global current_mode, active_video_path
    current_mode = 'webcam'
    active_video_path = None
    return render_template('index.html', mode='webcam')


@app.route('/default_video')
def default_video():
    global current_mode, active_video_path
    camera_thread.stop()
    current_mode = 'video'
    active_video_path = DEFAULT_VIDEO
    return render_template('index.html', mode='video')


@app.route('/reset')
def reset():
    global current_mode, active_video_path, latest_detections
    camera_thread.stop()
    current_mode = 'none'
    active_video_path = None
    latest_detections = []
    return render_template('index.html', mode='none')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_mode, active_video_path
    camera_thread.stop()

    if 'video' not in request.files:
        return render_template('index.html', mode=current_mode)

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', mode=current_mode)

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    active_video_path = save_path
    current_mode = 'video'
    return render_template('index.html', mode='video')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    global current_mode, active_video_path, latest_detections
    camera_thread.stop()

    if 'image' not in request.files:
        return render_template('index.html', mode=current_mode)

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', mode=current_mode)

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Lightweight inference inside no_grad
    with torch.no_grad():
        results = model(frame, conf=0.25, imgsz=640, verbose=False)

    annotated_frame, detections = draw_custom_boxes(frame, results)

    output_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    cv2.imwrite(output_path, annotated_frame)

    current_mode = 'image'
    active_video_path = None
    latest_detections = detections
    return render_template('index.html', mode='image',
                           image_result='/uploads/result.jpg',
                           detections=detections)


@app.route('/get_latest_detections')
def get_latest_detections():
    global latest_detections
    return jsonify(latest_detections)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True)