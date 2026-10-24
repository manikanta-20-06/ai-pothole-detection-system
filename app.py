import os
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
DEFAULT_VIDEO = os.path.join(BASE_DIR, 'sample_video.mp4')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained YOLO weights
model = YOLO(MODEL_PATH)

# Start by default on prerecorded video
current_mode = 'video'
active_video_path = DEFAULT_VIDEO

# Holds the latest live detection history
latest_detections = []


def calculate_severity(box_area, frame_area):
    """
    Classifies severity based on bbox area ratio:
    - Small  (< 1.5% area): Green  (Low)
    - Medium (1.5% - 5% area): Cyan/Blue (Medium)
    - Large  (> 5% area): Red    (High)
    """
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

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label_name = model.names.get(cls_id, 'Pothole')

        box_area = (x2 - x1) * (y2 - y1)
        severity, color = calculate_severity(box_area, frame_area)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw label
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

    # Update global detections so history table can show them
    if current_frame_detections:
        latest_detections = current_frame_detections

    return frame, current_frame_detections


def generate_frames():
    global current_mode, active_video_path

    if current_mode == 'webcam':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    elif current_mode == 'video' and active_video_path and os.path.exists(active_video_path):
        cap = cv2.VideoCapture(active_video_path)
    else:
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if current_mode == 'video':
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        results = model(frame, conf=0.25, verbose=False)
        annotated_frame, _ = draw_custom_boxes(frame, results)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
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
    return render_template('index.html', mode=current_mode)


@app.route('/default_video')
def default_video():
    global current_mode, active_video_path
    current_mode = 'video'
    active_video_path = DEFAULT_VIDEO
    return render_template('index.html', mode=current_mode)


@app.route('/reset')
def reset():
    global current_mode, active_video_path, latest_detections
    current_mode = 'none'
    active_video_path = None
    latest_detections = []
    return render_template('index.html', mode=current_mode)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_mode, active_video_path
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
    return render_template('index.html', mode=current_mode)


@app.route('/predict_image', methods=['POST'])
def predict_image():
    global current_mode, active_video_path, latest_detections
    if 'image' not in request.files:
        return render_template('index.html', mode=current_mode)

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', mode=current_mode)

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.25, verbose=False)
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
    """API endpoint to stream live detection updates to the history table"""
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