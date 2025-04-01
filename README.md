# Batter Pose Detector

A Python tool for detecting and analyzing baseball/softball batter poses using YOLOv8 and MediaPipe.

## Features

- Detects batters in video using YOLOv8
- Identifies the closest person to the bat
- Performs pose detection using MediaPipe
- Saves pose data in JSON format
- Supports video playback with frame rate control
- Saves annotated output video

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/batter_pose_detector.git
cd batter_pose_detector
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with a video file:

```bash
python detect_batter.py "path/to/your/video.mp4" [options]
```

### Options

- `--save`: Save the output video
- `--fps <number>`: Set playback frame rate (default: 30)
- `--model <path>`: Specify YOLO model path (default: yolov8x.pt)

### Example

```bash
python detect_batter.py "sample_video.mp4" --save --fps 30
```

## Output

- Annotated video with detected batters and pose landmarks
- JSON files containing pose data in `batter_dataset/poses/`
- Frame-by-frame detection information in the console

## Controls

During video playback:
- Space: Pause/Resume
- Left/Right arrows: Step frame
- Up/Down arrows: Adjust playback speed
- Q: Quit

## License

MIT License 