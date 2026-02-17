# 👁️ Third Eye – AI Navigation for the Visually Challenged

Third Eye is an AI-powered real-time obstacle detection and navigation system designed to assist visually challenged individuals.

It uses YOLOv8, computer vision, and intelligent grid-based path planning to detect obstacles and guide users in the safest direction.

## 🚀 Features
- Real-time obstacle detection using YOLOv8
- Intelligent center-line sensitive navigation
- GPU acceleration (CUDA support)
- Audio-based guidance
- Flask + SocketIO real-time communication
- Optimized grid path planning

## 🧠 Tech Stack
- Python
- YOLOv8
- OpenCV
- PyTorch
- Flask
- SocketIO
- NVIDIA GPU support

## 📦 Installation

### Clone repo
git clone https://github.com/yourusername/third-eye.git
cd third-eye

### Create virtual environment
python -m venv cyber-llm-env
cyber-llm-env\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Download YOLO weights
Download yolov8n.pt from:
https://github.com/ultralytics/ultralytics

Place inside project folder.

### Run
python app.py

## 🎯 Future Scope
- Voice assistant navigation
- Wearable device integration
- Mobile app
- GPS-based outdoor navigation
- Smart glasses integration

## ❤️ Impact
This project aims to improve independence and safety for visually impaired individuals.

