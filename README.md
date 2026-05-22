# Smoking Detection

Real-time smoking detection using the [Roboflow](https://roboflow.com) inference API. Detects faces and cigarettes in a webcam feed or video file, and triggers an alert when a cigarette is detected near a face.

## Features

- Real-time detection via webcam or video file
- Temporal frame buffering to reduce false positives
- Cooldown timer for smooth UI alerts
- Multi-threaded inference for low latency

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/introshia/smoking-detection.git
   cd smoking-detection
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your Roboflow API key:
   ```bash
   cp .env.example .env
   ```

## Usage

```bash
# Webcam (default)
python app.py

# Video file
python app.py --source video.mp4

# Image
python app.py --source image.jpg
```

Press `Q` to quit.
