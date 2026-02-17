from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch
import os

# Force PyTorch to use NVIDIA GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using NVIDIA GPU: {gpu_name}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

# Load YOLOv8 model
try:
    model = YOLO("yolov8n.pt")
    if str(device) == 'cuda':
        model.to(device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        print("Model loaded on NVIDIA GPU")
    else:
        print("Model loaded on CPU")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("YOLOv8 model loaded successfully!")

last_announcement = None
last_announcement_time = 0
announcement_delay = 3  # Reduced delay for more responsive commands

# Enhanced grid with sensitive center zones
NUM_ROWS = 3
NUM_COLS = 5

# Define center-sensitive zones
CENTER_LEFT_COL = 1   # Left of center line
CENTER_RIGHT_COL = 3  # Right of center line
CENTER_COL = 2        # True center

def get_grid_cell(x_center, y_bottom, zone_lines_y, zone_lines_x):
    """
    Determines the (row, col) of an object based on its position.
    Now with more sensitive center detection
    """
    # Determine Row (Distance)
    row = 0  # Far
    if y_bottom > zone_lines_y[0]:  # Below Far line
        row = 1  # Mid
    if y_bottom > zone_lines_y[1]:  # Below Mid line
        row = 2  # Near
    
    # Determine Column (Horizontal) - more sensitive to center
    col = 0  # Far-Left
    for i in range(len(zone_lines_x)):
        if x_center > zone_lines_x[i]:
            col = i + 1
    
    return row, col

def analyze_center_sensitivity(detections, zone_lines_x, width):
    """
    Enhanced analysis focusing on center sensitivity
    Returns: left_center_threat, right_center_threat, center_threat
    """
    left_center_threat = False
    right_center_threat = False 
    center_threat = False
    
    center_left_boundary = zone_lines_x[1]  # Left center line
    center_right_boundary = zone_lines_x[2]  # Right center line
    
    # Sensitivity margin - how close to the line to consider as threat
    sensitivity_margin = width * 0.05  # 5% of screen width
    
    for detection in detections:
        x_center = detection['x_center']
        
        # Check if object is touching or near left center line
        if abs(x_center - center_left_boundary) < sensitivity_margin:
            left_center_threat = True
        
        # Check if object is touching or near right center line  
        if abs(x_center - center_right_boundary) < sensitivity_margin:
            right_center_threat = True
            
        # Check if object is in true center
        if center_left_boundary < x_center < center_right_boundary:
            center_threat = True
    
    return left_center_threat, right_center_threat, center_threat

def calculate_zone_openness(cost_grid):
    """
    Calculate how open each zone is (left, center, right)
    Lower score = more open
    """
    # Left zone (columns 0-1)
    left_openness = np.sum(cost_grid[:, 0:2])
    
    # Center zone (column 2)
    center_openness = np.sum(cost_grid[:, 2])
    
    # Right zone (columns 3-4)
    right_openness = np.sum(cost_grid[:, 3:5])
    
    return {
        'left': left_openness,
        'center': center_openness, 
        'right': right_openness
    }

def get_navigation_announcement(cost_grid, obstacle_map, detections, zone_lines_x, width):
    """
    New navigation logic with center line sensitivity
    """
    # Analyze center line threats
    left_center_threat, right_center_threat, center_threat = analyze_center_sensitivity(
        detections, zone_lines_x, width
    )
    
    # Calculate zone openness
    openness = calculate_zone_openness(cost_grid)
    
    # Get obstacle names for announcements
    center_obstacles = list(set(
        obstacle_map[0][CENTER_COL] + 
        obstacle_map[1][CENTER_COL] + 
        obstacle_map[2][CENTER_COL]
    ))
    
    # --- Decision Logic ---
    
    # Case 1: Both center lines are threatened
    if left_center_threat and right_center_threat:
        # Choose the most open direction
        if openness['left'] < openness['right']:
            return "Obstacles on both sides. Move left for clearer path."
        else:
            return "Obstacles on both sides. Move right for clearer path."
    
    # Case 2: Only left center line is threatened
    elif left_center_threat:
        if openness['right'] < 5:  # Right is relatively clear
            return "Obstacle near left center. Shift slightly right."
        else:
            return "Obstacle on left. Carefully adjust right."
    
    # Case 3: Only right center line is threatened  
    elif right_center_threat:
        if openness['left'] < 5:  # Left is relatively clear
            return "Obstacle near right center. Shift slightly left."
        else:
            return "Obstacle on right. Carefully adjust left."
    
    # Case 4: Object in true center
    elif center_threat:
        if openness['left'] < openness['right']:
            return "Center blocked. Move left."
        else:
            return "Center blocked. Move right."
    
    # Case 5: Check overall path openness
    else:
        if openness['center'] < 3:  # Center is clear
            return "Path clear. Continue straight."
        else:
            # Center has some obstacles but not on lines
            if openness['left'] < openness['right']:
                return "Path ahead constrained. Suggest moving left."
            else:
                return "Path ahead constrained. Suggest moving right."

def generate_frames():
    global last_announcement, last_announcement_time
    
    # Configure camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam.")

    # Warm-up GPU
    print("Warming up GPU...")
    warmup_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    with torch.cuda.amp.autocast():
        _ = model(warmup_frame, verbose=False)
    torch.cuda.synchronize()
    print("GPU warm-up complete!")

    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        current_time = time.time()
        fps = frame_count / (current_time - start_time)
        
        height, width, _ = frame.shape

        # Define grid lines - making center lines more prominent
        zone_lines_y = [int(height * 0.60), int(height * 0.85)] 
        zone_lines_x = [int(width * (i / NUM_COLS)) for i in range(1, NUM_COLS)]
        
        # Draw grid with emphasis on center lines
        # Far and Near lines (yellow)
        cv2.line(frame, (0, zone_lines_y[0]), (width, zone_lines_y[0]), (0, 255, 255), 2)
        cv2.line(frame, (0, zone_lines_y[1]), (width, zone_lines_y[1]), (0, 165, 255), 2)
        
        # All vertical lines (yellow)
        for x in zone_lines_x:
            cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
        
        # Emphasize center lines with different color (red)
        center_left_line = zone_lines_x[1]  # Left center boundary
        center_right_line = zone_lines_x[2]  # Right center boundary
        
        cv2.line(frame, (center_left_line, 0), (center_left_line, height), (0, 0, 255), 3)
        cv2.line(frame, (center_right_line, 0), (center_right_line, height), (0, 0, 255), 3)
        
        # Highlight center zone
        center_zone_top = (center_left_line, 0)
        center_zone_bottom = (center_right_line, height)
        cv2.rectangle(frame, center_zone_top, center_zone_bottom, (0, 0, 255), 1)

        # Reset tracking arrays
        cost_grid = np.zeros((NUM_ROWS, NUM_COLS), dtype=int)
        obstacle_map = [[[] for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
        detections_list = []

        # YOLOv8 Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = model(frame, conf=0.5, verbose=False, imgsz=640)
        
        result = results[0]
        announcement = None
        
        # Process detections
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                x_center = (x1 + x2) / 2
                y_bottom = y2
                
                # Store detection for center sensitivity analysis
                detections_list.append({
                    'x_center': x_center,
                    'y_bottom': y_bottom,
                    'class_name': class_name,
                    'confidence': confidence
                })
                
                row, col = get_grid_cell(x_center, y_bottom, zone_lines_y, zone_lines_x)
                
                # Assign costs based on distance
                if row == 0:  # Far
                    cost = 1
                    color = (0, 255, 255)  # Yellow
                elif row == 1:  # Mid
                    cost = 3
                    color = (0, 165, 255)  # Orange
                else:  # Near
                    cost = 6
                    color = (0, 0, 255)    # Red
                
                cost_grid[row][col] += cost
                obstacle_map[row][col].append(class_name)
                
                # Draw bounding boxes
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Enhanced navigation with center sensitivity
            announcement = get_navigation_announcement(
                cost_grid, obstacle_map, detections_list, zone_lines_x, width
            )
        
        else:
            announcement = "Path clear. Continue straight ahead."
        
        # Display information
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Center-line Sensitive Mode", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show zone openness if we have detections
        if len(detections_list) > 0:
            openness = calculate_zone_openness(cost_grid)
            cv2.putText(frame, f"Openness L:{openness['left']} C:{openness['center']} R:{openness['right']}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # SocketIO announcement
        current_time = time.time()
        if announcement:
            if announcement != last_announcement or (current_time - last_announcement_time) > announcement_delay:
                socketio.emit('announcement', {'data': announcement})
                last_announcement = announcement
                last_announcement_time = current_time
        
        # Yield frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting center-line sensitive navigation system...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)