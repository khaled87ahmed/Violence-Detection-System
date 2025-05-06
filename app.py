from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from threading import Thread, Lock
import cv2
import os
import time
import uuid
from flask_socketio import SocketIO
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Initialize YOLO model
try:
    model = YOLO("violence_detection_model.pt")  # Update with your model path
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Global variables for camera handling
camera_lock = Lock()
camera = None
CAMERA_ON = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_violence(frame):
    """Use YOLO model to detect violence in a frame and check the class of detections"""
    if model is None:
        return False, frame
    
    try:
        results = model(frame, conf=0.50, verbose=False)
        
        for result in results:
            if len(result.boxes) > 0:  # If any detections
                for box in result.boxes:
                    class_id = int(box.cls)  # Get class ID
                    class_name = model.names[class_id]  # Get class name
                    
                    # Check if the detected class is "violence"
                    if class_name.lower() == "violence":
                        return True, result.plot()  # Return True and annotated frame
                    else:
                        return False, result.plot()  # Return False and annotated frame
        
        return False, frame  # No violence detected
    
    except Exception as e:
        print(f"Detection error: {e}")
        return False, frame

def process_video(input_path, output_path):
    """Process video file with violence detection"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        violence_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            violence, processed_frame = detect_violence(frame)
            if violence:
                violence_frames += 1

            out.write(processed_frame)
            current_frame += 1
            progress = (current_frame / frame_count) * 100
            socketio.emit('processing_progress', {'progress': progress})

        cap.release()
        out.release()
        
        violence_percentage = (violence_frames / frame_count) * 100 if frame_count > 0 else 0
        return True, violence_percentage
        
    except Exception as e:
        print(f"Video processing error: {e}")
        return False, 0

def generate_camera_frames():
    """Generate frames from webcam with violence detection"""
    global CAMERA_ON, camera
    with camera_lock:
        camera = cv2.VideoCapture(0)
        while CAMERA_ON:
            success, frame = camera.read()
            if not success:
                break
                
            _, processed_frame = detect_violence(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        camera.release()
        camera = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global CAMERA_ON
    if not CAMERA_ON:
        CAMERA_ON = True
        return jsonify({"status": "Webcam started"})
    return jsonify({"status": "Webcam already running"})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global CAMERA_ON
    CAMERA_ON = False
    return jsonify({"status": "Webcam stopped"})

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    video_id = str(uuid.uuid4())
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{video_id}_{filename}")
    
    try:
        file.save(input_path)
    except Exception as e:
        return jsonify({'error': f'File save failed: {str(e)}'}), 500

    def process_task():
        success, violence_percentage = process_video(input_path, output_path)
        if success:
            socketio.emit('processing_complete', {
                'original_video': f'/uploads/{video_id}_{filename}',
                'processed_video': f'/processed/processed_{video_id}_{filename}',
                'violence_percentage': round(violence_percentage, 2)
            })
        else:
            socketio.emit('processing_error', {
                'error': 'Video processing failed'
            })

    Thread(target=process_task).start()
    return jsonify({'status': 'processing_started'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'camera_active': CAMERA_ON
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)