# Batter Pose Detector

A Python tool for detecting and analyzing baseball/softball batter poses using YOLOv8 and MediaPipe.

## Features

- Detects batters in video using YOLOv8
- Identifies the closest person to the bat
- Performs pose detection using MediaPipe
- Detects bat position relative to hands
- Determines if hands are at shoulder level
- Supports video playback with frame rate control
- Saves annotated output video

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/batter_pose_detector.git
cd batter_pose_detector
```

2. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

## Usage

Run the script with a video file:

```bash
poetry run python detect_batter.py "path/to/your/video.mp4" [options]
```

### Options

- `--save`: Save the output video
- `--fps <number>`: Set playback frame rate (default: 30)
- `--model <path>`: Specify YOLO model path (default: yolov8x.pt)

### Example

```bash
poetry run python detect_batter.py "sample_video.mp4" --save --fps 30
```

## Output

- Annotated video with detected batters and pose landmarks
- Frame-by-frame detection information in the console including:
  - Bat position relative to hands
  - Hand height relative to shoulders
  - Detection confidence scores

## Controls

During video playback:
- Space: Pause/Resume
- Left/Right arrows: Step frame
- Up/Down arrows: Adjust playback speed
- Q: Quit

## License

MIT License 