import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO

def init_models(model_name='yolov8x.pt'):
    """Initialize YOLO and MediaPipe models."""
    print("Initializing models...")
    print(f"Loading YOLO model: {model_name}")
    yolo_model = YOLO(model_name)
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    return yolo_model, pose, mp_drawing

def setup_video_capture(video_path, save_output=False, playback_fps=15):
    """Setup video capture and writer."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None, None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {width}x{height} at {fps}fps, {total_frames} frames")
    print(f"Playing back at {playback_fps}fps")
    
    out = None
    if save_output:
        video_path = Path(video_path)
        output_path = video_path.parent / f"{video_path.stem}_batter{video_path.suffix}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, playback_fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    return cap, out, width, height, total_frames

def process_detections(results, frame):
    """Process YOLO detection results to find bats and people."""
    bat_bbox = None
    person_bboxes = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = r.names[cls]
            
            if class_name == "baseball bat" and conf > 0.5:
                # Draw bat detection box and label
                color = (0, 255, 0)  # Green for bats
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Found bat with confidence {conf:.2f}")
                bat_bbox = (x1, y1, x2, y2)
            elif class_name == "person" and conf > 0.5:
                print(f"Found person with confidence {conf:.2f}")
                person_bboxes.append((x1, y1, x2, y2))
    
    return bat_bbox, person_bboxes

def find_potential_batters(bat_bbox, person_bboxes):
    """Find potential batters based on overlap with bat."""
    if not bat_bbox or not person_bboxes:
        return None
        
    bat_x1, bat_y1, bat_x2, bat_y2 = bat_bbox
    bat_center_x = (bat_x1 + bat_x2) / 2
    bat_center_y = (bat_y1 + bat_y2) / 2
    
    potential_batters = []
    for person_idx, person_bbox in enumerate(person_bboxes):
        px1, py1, px2, py2 = person_bbox
        
        # Calculate overlap
        overlap_x = max(0, min(bat_x2, px2) - max(bat_x1, px1))
        overlap_y = max(0, min(bat_y2, py2) - max(bat_y1, py1))
        overlap_area = overlap_x * overlap_y
        
        bat_area = (bat_x2 - bat_x1) * (bat_y2 - bat_y1)
        overlap_ratio = overlap_area / bat_area
        
        if overlap_ratio > 0.5 or overlap_area > 0:
            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2
            distance = np.sqrt((person_center_x - bat_center_x)**2 + 
                             (person_center_y - bat_center_y)**2)
            
            potential_batters.append({
                'idx': person_idx,
                'distance': distance,
                'overlap_ratio': overlap_ratio
            })
    
    if potential_batters:
        potential_batters.sort(key=lambda x: (-x['overlap_ratio'], x['distance']))
        return potential_batters[0]['idx']
    
    return None

def process_pose(frame, person_bbox, pose, mp_drawing, width, height):
    """Process pose detection for the identified batter."""
    px1, py1, px2, py2 = person_bbox
    padding = 20
    roi_x1 = max(0, px1 - padding)
    roi_y1 = max(0, py1 - padding)
    roi_x2 = min(width, px2 + padding)
    roi_y2 = min(height, py2 + padding)
    
    # Extract ROI and process pose
    roi = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2RGB)
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    pose_results = pose.process(roi)
    
    if pose_results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            roi_frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi_frame
        cv2.putText(frame, "BATTER DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_frame_info(frame, frame_count, total_frames, height):
    """Draw frame counter and other information."""
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_person_boxes(frame, person_bboxes, batter_idx=None):
    """Draw bounding boxes for all detected people."""
    for idx, bbox in enumerate(person_bboxes):
        px1, py1, px2, py2 = bbox
        if idx == batter_idx:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(frame, "BATTER", (px1, py1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (px1, py1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def handle_keyboard_input(key, playback_fps):
    """Handle keyboard controls and return updated playback_fps."""
    if key == ord('q'):
        return None
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('+'):
        playback_fps = min(playback_fps + 5, 60)
        print(f"Playback speed increased to {playback_fps}fps")
    elif key == ord('-'):
        playback_fps = max(playback_fps - 5, 1)
        print(f"Playback speed decreased to {playback_fps}fps")
    return playback_fps

def process_frame(frame, model, pose, mp_drawing, width, height):
    """Process a single frame for detections and pose."""
    # Process detections
    results = model(frame)
    bat_bbox, person_bboxes = process_detections(results, frame)
    
    # Find and process potential batters
    batter_idx = None
    if bat_bbox and person_bboxes:
        batter_idx = find_potential_batters(bat_bbox, person_bboxes)
        if batter_idx is not None:
            process_pose(frame, person_bboxes[batter_idx], pose, mp_drawing, width, height)
    
    # Draw person boxes
    draw_person_boxes(frame, person_bboxes, batter_idx)
    
    return frame

def detect_batter_pose(video_path, save_output=False, playback_fps=15, model_name='yolov8x.pt'):
    """Main function to detect batter poses in video."""
    # Initialize models and video capture
    model, pose, mp_drawing = init_models(model_name)
    cap, out, width, height, total_frames = setup_video_capture(video_path, save_output, playback_fps)
    if cap is None:
        return
    
    frame_count = 0
    frame_delay = int(1000 / playback_fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\nProcessing frame {frame_count}/{total_frames}")
        
        # Draw frame counter
        draw_frame_info(frame, frame_count, total_frames, height)
        
        # Process frame
        frame = process_frame(frame, model, pose, mp_drawing, width, height)
        
        # Save and display frame
        if save_output:
            out.write(frame)
        
        cv2.imshow('Batter Pose Detection', frame)
        key = cv2.waitKey(frame_delay) & 0xFF
        
        # Handle keyboard controls
        new_fps = handle_keyboard_input(key, playback_fps)
        if new_fps is None:  # Quit signal
            break
        elif new_fps != playback_fps:
            playback_fps = new_fps
            frame_delay = int(1000 / playback_fps)
    
    # Cleanup
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect batter pose in video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--fps', type=int, default=15, help='Playback frame rate (default: 15)')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                      help='YOLO model to use (default: yolov8x.pt). Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt')
    args = parser.parse_args()
    
    detect_batter_pose(args.video_path, args.save, args.fps, args.model) 