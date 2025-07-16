from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from threading import Thread, Lock
import cv2
import os
import time
import uuid
from flask_socketio import SocketIO
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

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

EMAIL_SETTINGS = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'khaledahmed872003@gmail.com',
    'sender_password': 'czgf ylzg nema tdkl',
    'receiver_email': 'khaledahmed8720031@gmail.com'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_violence_alert(frame, timestamp):
    """Send an email alert with the violence detection frame"""
    if model is None:
        return False
    
    try:
        # Create the email message
        msg = MIMEMultipart('related')
        msg['Subject'] = "🚨 Violence Detected in Surveillance Feed"
        msg['From'] = EMAIL_SETTINGS['sender_email']
        msg['To'] = EMAIL_SETTINGS['receiver_email']
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Create HTML email content
        detection_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        html = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #d9534f; color: white; padding: 15px; text-align: center; border-radius: 5px 5px 0 0; }}
                    .content {{ padding: 20px; background-color: #f9f9f9; border-radius: 0 0 5px 5px; }}
                    .footer {{ margin-top: 20px; text-align: center; font-size: 12px; color: #777; }}
                    .image-container {{ margin: 15px 0; text-align: center; }}
                    .image-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
                    .btn {{ display: inline-block; padding: 10px 15px; background-color: #d9534f; color: white; text-decoration: none; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2>Violence Detection Alert</h2>
                    </div>
                    <div class="content">
                        <p>Our surveillance system has detected potential violent behavior in the video feed.</p>
                        
                        <div class="image-container">
                            <img src="cid:violence_frame" alt="Violence Detected Frame">
                        </div>
                        
                        <p><strong>Detection Details:</strong></p>
                        <ul>
                            <li><strong>Time:</strong> {detection_time}</li>
                            <li><strong>Location:</strong> Surveillance Camera 1</li>
                            <li><strong>Confidence:</strong> High probability</li>
                        </ul>
                        
                        <p>Please review this incident immediately and take appropriate action if necessary.</p>
                        
                    </div>
                    <div class="footer">
                        <p>This is an automated message. Please do not reply to this email.</p>
                        <p>&copy; {datetime.now().year} Security Surveillance System</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # Attach HTML content
        msg.attach(MIMEText(html, 'html'))
        
        # Attach the image
        image = MIMEImage(frame_bytes)
        image.add_header('Content-ID', '<violence_frame>')
        msg.attach(image)
        
        # Send the email
        with smtplib.SMTP(EMAIL_SETTINGS['smtp_server'], EMAIL_SETTINGS['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_SETTINGS['sender_email'], EMAIL_SETTINGS['sender_password'])
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_whatsapp_alert(detection_details):
    """Send WhatsApp alert using Facebook Graph API"""
    try:
        headers = {
            'Authorization': f'Bearer {WHATSAPP_CONFIG["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": WHATSAPP_CONFIG.get('recipient_number', '201553305791'),
            "type": "template",
            "template": {
                "name": WHATSAPP_CONFIG["template_name"],
                "language": {"code": "en_US"},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": detection_details['time']},
                            {"type": "text", "text": detection_details['location']},
                            {"type": "text", "text": detection_details['confidence']}
                        ]
                    }
                ]
            }
        }

        response = requests.post(
            WHATSAPP_CONFIG['api_url'],
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code != 200:
            print(f"WhatsApp API error: {response.text}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

def detect_violence(frame):
    """Use YOLO model to detect violence in a frame with custom confidence thresholds"""
    if model is None:
        return False, frame, None
    
    try:
        # Run detection with default confidence (we'll filter later)
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        original_frame = frame.copy()
        
        violence_detected = False
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    is_violence = class_name.lower() == "violence"
                    
                    # Apply different confidence thresholds
                    if (is_violence and confidence >= 0.7) or (not is_violence and confidence >= 0.3):
                        # Determine box color based on class
                        box_color = (0, 0, 255) if is_violence else (0, 255, 0)  # Red for violence, green for non-violence
                        
                        # Mark violence detected if we have a high-confidence violence detection
                        if is_violence and confidence >= 0.7:
                            violence_detected = True
                        
                        # Extract and draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness=10)
                        
                        # Add label
                        label = f"{class_name} {confidence:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        text_color = (255, 255, 255)
                        text_thickness = 3
                        text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
                        
                        # Background for text
                        text_bg_top_left = (x1, y1 - text_size[1] - 5)
                        text_bg_bottom_right = (x1 + text_size[0], y1)
                        cv2.rectangle(annotated_frame, text_bg_top_left, text_bg_bottom_right, box_color, -1)
                        
                        # Put the text on the frame
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), font, font_scale, text_color, text_thickness)
        
        return violence_detected, annotated_frame, original_frame if violence_detected else None
    
    except Exception as e:
        print(f"Detection error: {e}")
        return False, frame, None

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

            violence, processed_frame, original_frame = detect_violence(frame)
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
    """Generate frames from webcam with violence detection matching upload processing"""
    global CAMERA_ON, camera
    with camera_lock:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_email_time = 0  # To prevent too many emails
        email_cooldown = 100  # Seconds between emails
        
        while CAMERA_ON:
            success, frame = camera.read()
            if not success:
                break
                
            violence_detected, processed_frame, original_frame = detect_violence(frame)
            
            current_time = time.time()
            if violence_detected and (current_time - last_email_time) > email_cooldown:
                # Send email in a separate thread to avoid blocking
                Thread(target=send_violence_alert, args=(processed_frame, current_time)).start()
                last_email_time = current_time
                socketio.emit('violence_detected', {
                    'timestamp': current_time,
                    'message': 'Violence detected! Email alert sent.'
                })
            
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




