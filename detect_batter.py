import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import os
import json
from datetime import datetime

def detect_batter_pose(video_path, save_output=False, playback_fps=15, model_name='yolov8x.pt'):
    print("Initializing models...")
    # Initialize YOLO model for bat and person detection
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)  # using larger, more accurate model
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    mp_drawing = mp.solutions.drawing_utils

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {width}x{height} at {fps}fps, {total_frames} frames")
    print(f"Playing back at {playback_fps}fps")

    # Setup video writer if saving output
    if save_output:
        video_path = Path(video_path)
        output_path = video_path.parent / f"{video_path.stem}_batter{video_path.suffix}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, playback_fps, (width, height))
        print(f"Saving output to: {output_path}")

    frame_count = 0
    last_save_time = 0
    save_interval = 1.0  # Save every 1 second
    frame_delay = int(1000 / playback_fps)  # Convert fps to milliseconds

    # Create dataset directory
    dataset_dir = "batter_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "poses"), exist_ok=True)

    def save_pose_data(frame, pose_results, roi_coords, frame_number, timestamp):
        """Save frame and pose data to dataset."""
        # Generate unique filename based on timestamp and frame number
        base_filename = f"{timestamp}_{frame_number:06d}"
        
        # Save frame
        frame_path = os.path.join(dataset_dir, "frames", f"{base_filename}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Extract pose data
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            pose_data = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "roi_coords": {
                    "x1": roi_coords[0],
                    "y1": roi_coords[1],
                    "x2": roi_coords[2],
                    "y2": roi_coords[3]
                },
                "landmarks": {
                    "left_hip": {"x": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                "y": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                "z": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z,
                                "visibility": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility},
                    "right_hip": {"x": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 "y": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                 "z": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z,
                                 "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility},
                    "left_shoulder": {"x": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     "y": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                     "z": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                                     "visibility": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility},
                    "right_shoulder": {"x": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      "y": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                      "z": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
                                      "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility},
                    "left_elbow": {"x": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  "y": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                  "z": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                                  "visibility": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility},
                    "right_elbow": {"x": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   "y": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                   "z": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,
                                   "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility},
                    "left_wrist": {"x": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  "y": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                  "z": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                                  "visibility": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility},
                    "right_wrist": {"x": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   "y": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                   "z": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z,
                                   "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility}
                }
            }
            
            # Save pose data
            pose_path = os.path.join(dataset_dir, "poses", f"{base_filename}.json")
            with open(pose_path, 'w') as f:
                json.dump(pose_data, f, indent=2)
            
            print(f"Saved pose data to {pose_path}")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every frame
        frame_count += 1
        print(f"\nProcessing frame {frame_count}/{total_frames}")
        
        # Add frame number to display
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Run YOLO detection
        results = model(frame)
        
        # Process YOLO results
        bat_detected = False
        bat_bbox = None
        person_bboxes = []
        
        # Draw all detections and look for bats and people
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Draw all detections with confidence
                color = (0, 255, 0) if class_name == "baseball bat" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if class_name == "baseball bat" and conf > 0.3:  # Lower threshold for bat detection
                    print(f"Found bat with confidence {conf:.2f}")
                    bat_detected = True
                    bat_bbox = (x1, y1, x2, y2)
                elif class_name == "person" and conf > 0.5:
                    print(f"Found person with confidence {conf:.2f}")
                    person_bboxes.append((x1, y1, x2, y2))

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only process pose if we have a bat and people detected
        if bat_detected and person_bboxes:
            # Find the person closest to the bat
            bat_x1, bat_y1, bat_x2, bat_y2 = bat_bbox
            bat_center_x = (bat_x1 + bat_x2) / 2
            bat_center_y = (bat_y1 + bat_y2) / 2
            
            # First, find all people whose bounding box contains or is very close to the bat
            potential_batters = []
            for person_idx, person_bbox in enumerate(person_bboxes):
                px1, py1, px2, py2 = person_bbox
                
                # Calculate how much the bat overlaps with the person's box
                overlap_x = max(0, min(bat_x2, px2) - max(bat_x1, px1))
                overlap_y = max(0, min(bat_y2, py2) - max(bat_y1, py1))
                overlap_area = overlap_x * overlap_y
                
                # Calculate bat area
                bat_area = (bat_x2 - bat_x1) * (bat_y2 - bat_y1)
                
                # Calculate overlap ratio
                overlap_ratio = overlap_area / bat_area
                
                # If bat is mostly within person's box (overlap > 50%) or very close
                if overlap_ratio > 0.5 or overlap_area > 0:
                    # Calculate distance between bat center and person center
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2
                    distance = np.sqrt((person_center_x - bat_center_x)**2 + 
                                     (person_center_y - bat_center_y)**2)
                    
                    potential_batters.append({
                        'idx': person_idx,
                        'distance': distance,
                        'overlap_ratio': overlap_ratio
                    })
            
            # If we found potential batters, find the best one
            if potential_batters:
                # Sort by overlap ratio (highest first) and distance (lowest first)
                potential_batters.sort(key=lambda x: (-x['overlap_ratio'], x['distance']))
                closest_person_idx = potential_batters[0]['idx']
                print(f"Found {len(potential_batters)} potential batters")
                print(f"Selected batter at index {closest_person_idx} with overlap ratio {potential_batters[0]['overlap_ratio']:.2f}")
            else:
                # If no one is close enough to the bat, use the old distance-based method
                closest_person_idx = None
                min_distance = float('inf')
                
                for person_idx, person_bbox in enumerate(person_bboxes):
                    px1, py1, px2, py2 = person_bbox
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2
                    
                    distance = np.sqrt((person_center_x - bat_center_x)**2 + 
                                     (person_center_y - bat_center_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_person_idx = person_idx
                
                if closest_person_idx is not None:
                    print(f"No close batters found, using distance-based method")
                    print(f"Found person closest to bat (distance: {min_distance:.2f})")
                    print(f"Batter identified at index {closest_person_idx} with distance {min_distance:.2f}")
            
            # Get the batter's bounding box
            px1, py1, px2, py2 = person_bboxes[closest_person_idx]
            
            # Extract the batter's region with some padding
            padding = 20
            roi_x1 = max(0, px1 - padding)
            roi_y1 = max(0, py1 - padding)
            roi_x2 = min(width, px2 + padding)
            roi_y2 = min(height, py2 + padding)
            
            # Extract the ROI
            roi = rgb_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Create a temporary frame for the ROI
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            # Run pose detection only on the ROI
            pose_results = pose.process(roi)
            
            if pose_results.pose_landmarks:
                # Draw pose landmarks on the ROI frame
                mp_drawing.draw_landmarks(
                    roi_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Get all landmarks
                landmarks = pose_results.pose_landmarks.landmark
                
                # Get key points for batter analysis
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Calculate positions in ROI coordinates
                roi_width = roi_x2 - roi_x1
                roi_height = roi_y2 - roi_y1
                
                hip_x = int((left_hip.x + right_hip.x) * roi_width / 2)
                hip_y = int((left_hip.y + right_hip.y) * roi_height / 2)
                shoulder_x = int((left_shoulder.x + right_shoulder.x) * roi_width / 2)
                shoulder_y = int((left_shoulder.y + right_shoulder.y) * roi_height / 2)
                left_elbow_x = int(left_elbow.x * roi_width)
                left_elbow_y = int(left_elbow.y * roi_height)
                right_elbow_x = int(right_elbow.x * roi_width)
                right_elbow_y = int(right_elbow.y * roi_height)
                
                # Draw key points on ROI frame
                cv2.circle(roi_frame, (hip_x, hip_y), 5, (0, 255, 0), -1)
                cv2.circle(roi_frame, (shoulder_x, shoulder_y), 5, (0, 0, 255), -1)
                cv2.circle(roi_frame, (left_elbow_x, left_elbow_y), 5, (255, 0, 0), -1)
                cv2.circle(roi_frame, (right_elbow_x, right_elbow_y), 5, (255, 0, 0), -1)
                
                # Copy the ROI frame back to the main frame
                frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi_frame
                
                # Add text with coordinates and "Batter" label
                cv2.putText(frame, "BATTER DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save pose data if enough time has passed since last save
                current_time = frame_count / fps
                if current_time - last_save_time >= save_interval:
                    save_pose_data(frame, pose_results, (roi_x1, roi_y1, roi_x2, roi_y2), 
                                 frame_count, datetime.now().strftime("%Y%m%d_%H%M%S"))
                    last_save_time = current_time
            
            # Draw bounding boxes for all people
            for person_idx, person_bbox in enumerate(person_bboxes):
                px1, py1, px2, py2 = person_bbox
                if person_idx == closest_person_idx:
                    # Draw batter's box in red with "BATTER" label
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(frame, f"BATTER", (px1, py1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw other people's boxes in blue
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Person", (px1, py1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save frame if requested
        if save_output:
            out.write(frame)

        # Display the frame with controlled frame rate
        cv2.imshow('Batter Pose Detection', frame)
        key = cv2.waitKey(frame_delay) & 0xFF
        
        # Handle keyboard controls
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Space to pause
            cv2.waitKey(0)
        elif key == ord('+'):  # Increase playback speed
            playback_fps = min(playback_fps + 5, 60)
            frame_delay = int(1000 / playback_fps)
            print(f"Playback speed increased to {playback_fps}fps")
        elif key == ord('-'):  # Decrease playback speed
            playback_fps = max(playback_fps - 5, 1)
            frame_delay = int(1000 / playback_fps)
            print(f"Playback speed decreased to {playback_fps}fps")

    # Clean up
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