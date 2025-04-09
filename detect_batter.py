import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import time
import math

# Feature flag to control batter detection method
USE_POSITION_DETECTION = True  # True = use position model, False = use closest-to-bat

def init_models(model_name='yolov8x.pt'):
    """Initialize YOLO and MediaPipe models."""
    print("Initializing models...")
    print(f"Loading YOLO model: {model_name}")
    yolo_model = YOLO(model_name)
    
    # Add position model for player detection
    print("Loading position model: weights-pos-v7.pt")
    position_model = YOLO('weights-pos-v7.pt')
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    return yolo_model, position_model, pose, mp_drawing

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

# commented out because unneeded, will remove when new method is working
# def process_detections(results, frame):
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
    
    # Calculate bat's longest dimension
    bat_width = bat_x2 - bat_x1
    bat_height = bat_y2 - bat_y1
    bat_longest_dim = max(bat_width, bat_height)
    
    potential_batters = []
    for person_idx, person_bbox in enumerate(person_bboxes):
        px1, py1, px2, py2 = person_bbox
        
        # Calculate person's dimensions
        person_width = px2 - px1
        person_height = py2 - py1
        person_longest_dim = max(person_width, person_height)
        
        # Skip if person is not at least 1.5x the bat's longest dimension. 
        # this is so we dont pick up fielders who are far away from the bat
        if person_longest_dim < bat_longest_dim * 1.5:
            continue
        
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

def roi_to_full_frame(roi_x, roi_y, roi_x1, roi_y1, roi_width, roi_height):
    """Convert ROI coordinates to full frame coordinates.
    
    Args:
        roi_x: x coordinate in ROI space (0-1 normalized)
        roi_y: y coordinate in ROI space (0-1 normalized)
        roi_x1: starting x coordinate of ROI in full frame
        roi_y1: starting y coordinate of ROI in full frame
        roi_width: width of ROI
        roi_height: height of ROI
        
    Returns:
        Tuple of (full_frame_x, full_frame_y)
    """
    # Convert from normalized (0-1) to ROI pixels
    roi_pixel_x = roi_x * roi_width
    roi_pixel_y = roi_y * roi_height
    
    # Convert to full frame coordinates
    full_frame_x = roi_pixel_x + roi_x1
    full_frame_y = roi_pixel_y + roi_y1
    
    return full_frame_x, full_frame_y

def process_pose(frame, roi, bat_bbox, pose, mp_drawing, frame_count):
    """Process pose detection for the identified batter.
    
    Args:
        frame: Full video frame
        roi: Tuple of (x1, y1, x2, y2) for region of interest
        bat_bbox: Tuple of (x1, y1, x2, y2) for bat bounding box
        pose: MediaPipe pose object
        mp_drawing: MediaPipe drawing utilities
        frame_count: Current frame number
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    
    # Extract ROI and process pose
    roi_rgb = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2RGB)
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    pose_results = pose.process(roi_rgb)
    
    if pose_results.pose_landmarks:
        # Check if batter is in stance
        if is_batting_stance(bat_bbox, pose_results, roi, frame.shape[0]):
            print("Batter is in stance!")
            cv2.putText(frame, "BATTER IN STANCE", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save the frame only if 100 frames have passed since last save
            if not hasattr(process_pose, 'last_stance_frame') or frame_count - process_pose.last_stance_frame >= 100:
                output_dir = Path("stance_frames")
                output_dir.mkdir(exist_ok=True)
                frame_path = output_dir / f"stance_{int(time.time())}.jpg"
                cv2.imwrite(str(frame_path), frame)
                print(f"Saved stance frame to {frame_path}")
                process_pose.last_stance_frame = frame_count
        
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

def is_bat_above_hands(bat_bbox, full_frame_left_hand, full_frame_right_hand):
    """Check if the bat's center is above both hands.
    
    Args:
        bat_bbox: Tuple of (x1, y1, x2, y2) for bat bounding box in full frame
        full_frame_left_hand: Tuple of (x, y) for left hand in full frame
        full_frame_right_hand: Tuple of (x, y) for right hand in full frame
        
    Returns:
        bool: True if bat's center is above both hands, False otherwise
    """
    if not bat_bbox or not full_frame_left_hand or not full_frame_right_hand:
        return False
        
    bat_x1, bat_y1, bat_x2, bat_y2 = bat_bbox
    left_hand_x, left_hand_y = full_frame_left_hand
    right_hand_x, right_hand_y = full_frame_right_hand
    
    # Calculate bat's center y-coordinate
    bat_center_y = (bat_y1 + bat_y2) / 2
    
    # Print debug info
    print(f"Bat center Y: {bat_center_y:.1f}")
    print(f"Left hand Y: {left_hand_y:.1f}")
    print(f"Right hand Y: {right_hand_y:.1f}")
    
    # Check if the bat's center is above both hands
    return bat_center_y < left_hand_y and bat_center_y < right_hand_y

def is_bat_in_hands(bat_bbox, full_frame_left_hand, full_frame_right_hand, frame_width, frame_height):
    """Check if the bat is being held at the bottom by the batter's hands.
    
    Args:
        bat_bbox: Tuple of (x1, y1, x2, y2) for bat bounding box in full frame
        full_frame_left_hand: Tuple of (x, y) for left hand in full frame
        full_frame_right_hand: Tuple of (x, y) for right hand in full frame
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
        
    Returns:
        bool: True if bat is being held at the bottom by hands, False otherwise
    """
    if not bat_bbox or not full_frame_left_hand or not full_frame_right_hand:
        return False
        
    # Get bat bottom (highest y-coordinate since y increases downward)
    bat_x1, bat_y1, bat_x2, bat_y2 = bat_bbox
    bat_bottom_y = max(bat_y1, bat_y2)    # Bottom of bat
    
    # Get hands' positions
    left_x, left_y = full_frame_left_hand
    right_x, right_y = full_frame_right_hand
    
    # Check if hands are close together and near bat bottom
    hand_distance = abs(left_x - right_x)
    max_hand_distance = frame_width * 0.07  # 5% of frame width
    max_vertical_distance = frame_height * 0.09  # 5% of frame height
    
    # Log the actual distances in pixels
    print(f"Hand distance: {hand_distance:.1f}px (max: {max_hand_distance:.1f}px)")
    print(f"Vertical distance: {abs(max(left_y, right_y) - bat_bottom_y):.1f}px (max: {max_vertical_distance:.1f}px)")
    
    return (hand_distance <= max_hand_distance and  # Hands are close together
            abs(max(left_y, right_y) - bat_bottom_y) <= max_vertical_distance)  # Hands are near bat bottom

def are_hands_at_shoulders(full_frame_left_hand, full_frame_right_hand, full_frame_left_shoulder, full_frame_right_shoulder, frame_height):
    """Check if the batter's hands are at shoulder level.
    
    Args:
        full_frame_left_hand: Tuple of (x, y) for left hand in full frame
        full_frame_right_hand: Tuple of (x, y) for right hand in full frame
        full_frame_left_shoulder: Tuple of (x, y) for left shoulder in full frame
        full_frame_right_shoulder: Tuple of (x, y) for right shoulder in full frame
        frame_height: Height of the frame in pixels
        
    Returns:
        bool: True if hands are at shoulder level, False otherwise
    """
    if not all([full_frame_left_hand, full_frame_right_hand, 
                full_frame_left_shoulder, full_frame_right_shoulder]):
        return False
        
    # Get shoulder y-coordinates (average of left and right)
    left_shoulder_y = full_frame_left_shoulder[1]
    right_shoulder_y = full_frame_right_shoulder[1]
    shoulder_level = (left_shoulder_y + right_shoulder_y) / 2
    
    # Get hand y-coordinates
    left_hand_y = full_frame_left_hand[1]
    right_hand_y = full_frame_right_hand[1]
    
    # Maximum vertical distance allowed between hands and shoulders
    # Using 2% of frame height as the threshold
    max_vertical_distance = frame_height * 0.02
    
    # Print debug info
    print(f"Shoulder level Y: {shoulder_level:.1f}")
    print(f"Left hand Y: {left_hand_y:.1f}")
    print(f"Right hand Y: {right_hand_y:.1f}")
    print(f"Max allowed distance: {max_vertical_distance:.1f}")
    
    # Check if both hands are within the allowed distance from shoulder level
    return (abs(left_hand_y - shoulder_level) <= max_vertical_distance and
            abs(right_hand_y - shoulder_level) <= max_vertical_distance)

def are_knees_bent(full_frame_left_knee, full_frame_right_knee, full_frame_left_hip, full_frame_right_hip):
    """Check if the batter's knees are bent in a proper batting stance.
    
    Args:
        full_frame_left_knee: Tuple of (x, y) for left knee in full frame, or None if not visible
        full_frame_right_knee: Tuple of (x, y) for right knee in full frame, or None if not visible
        full_frame_left_hip: Tuple of (x, y) for left hip in full frame, or None if not visible
        full_frame_right_hip: Tuple of (x, y) for right hip in full frame, or None if not visible
        
    Returns:
        bool: True if knees are bent appropriately, False if not bent or if legs are not visible
    """
    # If we can't see any complete leg (knee + hip), return True
    if not ((full_frame_left_knee and full_frame_left_hip) or 
            (full_frame_right_knee and full_frame_right_hip)):
        print("No complete legs visible - assuming proper knee bend")
        return True
        
    # Define minimum angle threshold (in degrees)
    min_angle = 15  # Minimum angle between hip and knee
        
    # Check left leg if visible
    if full_frame_left_knee and full_frame_left_hip:
        # Calculate angle between hip and knee
        dx = full_frame_left_knee[0] - full_frame_left_hip[0]
        dy = full_frame_left_knee[1] - full_frame_left_hip[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        print(f"Left knee angle: {angle:.1f}°")
        
        if angle >= min_angle:
            return True
            
    # Check right leg if visible
    if full_frame_right_knee and full_frame_right_hip:
        # Calculate angle between hip and knee
        dx = full_frame_right_knee[0] - full_frame_right_hip[0]
        dy = full_frame_right_knee[1] - full_frame_right_hip[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        print(f"Right knee angle: {angle:.1f}°")
        
        if angle >= min_angle:
            return True
            
    # If we can see at least one complete leg but it's not bent properly, return False
    if ((full_frame_left_knee and full_frame_left_hip) or 
        (full_frame_right_knee and full_frame_right_hip)):
        print("Visible leg(s) not bent properly")
        return False
        
    return True

def is_batting_stance(bat_bbox, pose_results, roi, frame_height):
    """Check if the batter is in a proper batting stance.
    
    A proper batting stance requires:
    1. The bat is being held in the hands
    2. The bat is above the hands
    3. The hands are at shoulder level
    
    Args:
        bat_bbox: Tuple of (x1, y1, x2, y2) for bat bounding box in full frame
        pose_results: MediaPipe pose detection results
        roi: Tuple of (x1, y1, x2, y2) for region of interest
        frame_height: Height of the frame in pixels
        
    Returns:
        bool: True if all conditions for a proper batting stance are met, False otherwise
    """
    if not pose_results.pose_landmarks:
        return False
        
    # Extract ROI coordinates
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
        
    # Get hand and shoulder landmarks
    left_wrist = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Convert to full frame coordinates
    full_frame_left_hand = roi_to_full_frame(
        left_wrist.x, left_wrist.y,
        roi_x1, roi_y1,
        roi_width, roi_height
    )
    full_frame_right_hand = roi_to_full_frame(
        right_wrist.x, right_wrist.y,
        roi_x1, roi_y1,
        roi_width, roi_height
    )
    full_frame_left_shoulder = roi_to_full_frame(
        left_shoulder.x, left_shoulder.y,
        roi_x1, roi_y1,
        roi_width, roi_height
    )
    full_frame_right_shoulder = roi_to_full_frame(
        right_shoulder.x, right_shoulder.y,
        roi_x1, roi_y1,
        roi_width, roi_height
    )
    
    # Perform the checks using full frame coordinates
    return (is_bat_in_hands(bat_bbox, full_frame_left_hand, full_frame_right_hand, roi_width, roi_height) and
            is_bat_above_hands(bat_bbox, full_frame_left_hand, full_frame_right_hand) and
            are_hands_at_shoulders(full_frame_left_hand, full_frame_right_hand,
                                 full_frame_left_shoulder, full_frame_right_shoulder,
                                 frame_height))

def process_frame(frame, model, position_model, pose, mp_drawing, width, height, frame_count):
    """Process a single frame for detections and pose."""
    # Run YOLO detection
    results = model(frame)
    
    # Process YOLO results to get bat
    bat_bbox = None
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = r.names[cls]
            
            if class_name == "baseball bat" and conf > 0.5:
                bat_bbox = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Found bat with confidence {conf:.2f}")
                break

    # Get batter bbox
    batter_bbox = None
    if USE_POSITION_DETECTION:
        # Use position model
        position_results = position_model(frame)
        for r in position_results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if r.names[cls] == "Batter" and conf > 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    batter_bbox = (x1, y1, x2, y2)
                    print(f"BATTER DETECTED BY POSITION! Confidence: {conf:.2f}")
                    break
            if batter_bbox:
                break
    else:
        # Use old method - find closest person to bat
        person_bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if r.names[cls] == "person" and conf > 0.5:
                    person_bboxes.append((x1, y1, x2, y2))
                    print(f"Found person with confidence {conf:.2f}")
        
        if bat_bbox and person_bboxes:
            batter_idx = find_potential_batters(bat_bbox, person_bboxes)
            if batter_idx is not None:
                batter_bbox = person_bboxes[batter_idx]

    # Process pose if we found both bat and batter
    if bat_bbox and batter_bbox:
        # Calculate ROI
        px1, py1, px2, py2 = batter_bbox
        padding = 20
        roi = (
            max(0, px1 - padding),
            max(0, py1 - padding),
            min(width, px2 + padding),
            min(height, py2 + padding)
        )
        process_pose(frame, roi, bat_bbox, pose, mp_drawing, frame_count)

        # Draw batter box
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(frame, "BATTER", (px1, py1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def detect_batter_pose(video_path, save_output=False, playback_fps=15, model_name='yolov8x.pt', start_frame=0):
    """Main function to detect batter poses in video.
    
    Args:
        video_path: Path to the video file
        save_output: Whether to save the output video
        playback_fps: Frame rate for playback
        model_name: Name of the YOLO model to use
        start_frame: Frame number to start processing from (0-based)
    """
    # Initialize models and video capture
    model, position_model, pose, mp_drawing = init_models(model_name)
    cap, out, width, height, total_frames = setup_video_capture(video_path, save_output, playback_fps)
    if cap is None:
        return
    
    # Set the starting frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
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
        frame = process_frame(frame, model, position_model, pose, mp_drawing, width, height, frame_count)
        
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
    parser.add_argument('--start-frame', type=int, default=0,
                      help='Frame number to start processing from (0-based, default: 0)')
    args = parser.parse_args()
    
    detect_batter_pose(args.video_path, args.save, args.fps, args.model, args.start_frame) 