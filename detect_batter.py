import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import time
import math
import os

# Feature flag to control batter detection method
USE_POSITION_DETECTION = True  # True = use position model, False = use closest-to-bat

def init_models():
    """Initialize YOLO models for bat and player detection"""
    print("Initializing models...")
    # Initialize YOLO model for bat detection
    model = YOLO('yolov8x.pt')  # Changed back to yolov8x.pt
    model.fuse()
    
    # Initialize position model for player detection
    position_model = YOLO('weights-pos-v7.pt')
    position_model.fuse()
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    return model, position_model, pose, mp_drawing

def setup_video_capture(video_path, save_output=False):
    """Initialize video capture and output writer if needed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create cap_info dictionary
    cap_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }
    
    # Initialize output writer if save_output is True
    out = None
    if save_output:
        output_path = os.path.splitext(video_path)[0] + '_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap_info['output_path'] = output_path
    
    print(f"Video properties: {width}x{height} at {fps}fps, {total_frames} frames")
    return cap, out, cap_info


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

def roi_to_full_frame(x, y, roi_x1, roi_y1, roi_width, roi_height):
    """Convert coordinates from ROI to full frame coordinates.
    
    Args:
        x, y: Coordinates in ROI (0-1 normalized)
        roi_x1, roi_y1: Top-left corner of ROI in full frame
        roi_width, roi_height: Dimensions of ROI
        
    Returns:
        Tuple of (x, y) in full frame coordinates
    """
    # Convert normalized coordinates to ROI pixel coordinates
    roi_pixel_x = x * roi_width
    roi_pixel_y = y * roi_height
    
    # Convert to full frame coordinates
    full_frame_x = roi_x1 + roi_pixel_x
    full_frame_y = roi_y1 + roi_pixel_y  
    
    # Detailed debug output
    print(f"\nROI to Full Frame Debug:")
    print(f"Input normalized coords: ({x:.3f}, {y:.3f})")
    print(f"ROI dimensions: {roi_width}x{roi_height}")
    print(f"ROI offset: ({roi_x1}, {roi_y1})")
    print(f"ROI pixel coords: ({roi_pixel_x:.1f}, {roi_pixel_y:.1f})")
    print(f"Full frame coords: ({full_frame_x:.1f}, {full_frame_y:.1f})")
    print(f"Frame height: {roi_y1 + roi_height}")
    print(f"Distance from top: {roi_pixel_y:.1f}px")
    print(f"Distance from bottom: {(roi_height - roi_pixel_y):.1f}px")
    
    return (full_frame_x, full_frame_y)

def process_pose(frame, roi, bat_bbox, pose, mp_drawing):
    """Process pose detection for the identified batter."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    
    # Extract ROI and process pose
    roi_rgb = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2RGB)
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    pose_results = pose.process(roi_rgb)
    
    if pose_results.pose_landmarks:
        # Debug raw MediaPipe coordinates
        left_wrist = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        print(f"\nRaw MediaPipe coordinates:")
        print(f"Left wrist: ({left_wrist.x:.3f}, {left_wrist.y:.3f})")
        print(f"Right wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f})")
        
        # Check if batter is in stance
        if bat_bbox is not None and is_batting_stance(bat_bbox, pose_results, roi, frame.shape[0], frame.shape[1], frame):
            print("Batter is in stance!")
            cv2.putText(frame, "BATTER IN STANCE", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            roi_frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi_frame

def draw_frame_info(frame, frame_count, total_frames, height):
    """Draw frame counter and other information."""
    # Draw frame counter
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    
    # Calculate hand distance and check if it's within reasonable range
    hand_distance = abs(left_x - right_x)
    max_hand_distance = frame_width * 0.05  # 5% of frame width
    max_vertical_distance = frame_height * 0.09  # 9% of frame height
    
    # Debug output for coordinates
    print(f"\nBat coordinates:")
    print(f"Bat bbox: ({bat_x1}, {bat_y1}) to ({bat_x2}, {bat_y2})")
    print(f"Bat bottom Y: {bat_bottom_y}")
    print(f"Left hand: ({left_x}, {left_y})")
    print(f"Right hand: ({right_x}, {right_y})")
    
    # Log the actual distances in pixels
    print(f"Hand distance: {hand_distance:.1f}px (max: {max_hand_distance:.1f}px)")
    print(f"Vertical distance: {abs(min(left_y, right_y) - bat_bottom_y):.1f}px (max: {max_vertical_distance:.1f}px)")
    
    return (hand_distance <= max_hand_distance and  # Hands are close together
            abs(min(left_y, right_y) - bat_bottom_y) <= max_vertical_distance)  # Higher hand is near bat bottom

def are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder):
    """Check if the batter's hands are at shoulder level.
    
    Args:
        left_hand: Tuple of (x, y) for left hand in normalized coordinates (0-1)
        right_hand: Tuple of (x, y) for right hand in normalized coordinates (0-1)
        left_shoulder: Tuple of (x, y) for left shoulder in normalized coordinates (0-1)
        right_shoulder: Tuple of (x, y) for right shoulder in normalized coordinates (0-1)
        
    Returns:
        bool: True if hands are at shoulder level, False otherwise
    """
    if not all([left_hand, right_hand, left_shoulder, right_shoulder]):
        return False
        
    # Get shoulder y-coordinates (average of left and right)
    left_shoulder_y = left_shoulder[1]
    right_shoulder_y = right_shoulder[1]
    shoulder_level = (left_shoulder_y + right_shoulder_y) / 2
    
    # Get hand y-coordinates
    left_hand_y = left_hand[1]
    right_hand_y = right_hand[1]
    
    # Maximum vertical distance allowed between hands and shoulders
    # Using 10% of normalized height as the threshold
    max_vertical_distance = 0.10
    
    # Print debug info
    print(f"Shoulder level Y: {shoulder_level:.3f}")
    print(f"Left hand Y: {left_hand_y:.3f}")
    print(f"Right hand Y: {right_hand_y:.3f}")
    print(f"Max allowed distance: {max_vertical_distance:.3f}")
    
    # Check if both hands are within the allowed distance from shoulder level
    return (abs(left_hand_y - shoulder_level) <= max_vertical_distance and
            abs(right_hand_y - shoulder_level) <= max_vertical_distance)

def are_knees_bent(left_knee, right_knee, left_hip, right_hip):
    # If no complete legs are visible, assume proper stance
    if not (left_knee and left_hip) and not (right_knee and right_hip):
        return True
        
    min_angle = 12  # Minimum angle from vertical
        
    # Check left leg if visible
    left_leg_bent = False
    if left_knee and left_hip:
        # Measure angle from vertical (arctan of dx/dy)
        dx = abs(left_knee.x - left_hip.x)
        dy = abs(left_knee.y - left_hip.y)
        if dy == 0:  # Horizontal leg (90 degrees from vertical)
            angle = 90
        else:
            angle = math.degrees(math.atan(dx/dy))
        print(f"Left leg angle: {angle:.1f}°")
        if angle >= min_angle:
            left_leg_bent = True
            
    # Check right leg if visible
    right_leg_bent = False
    if right_knee and right_hip:
        # Same measurement for right leg
        dx = abs(right_knee.x - right_hip.x)
        dy = abs(right_knee.y - right_hip.y)
        if dy == 0:  # Horizontal leg (90 degrees from vertical)
            angle = 90
        else:
            angle = math.degrees(math.atan(dx/dy))
        print(f"Right leg angle: {angle:.1f}°")
        if angle >= min_angle:
            right_leg_bent = True
            
    # If both legs are visible, both must be bent
    if (left_knee and left_hip) and (right_knee and right_hip):
        return left_leg_bent and right_leg_bent
    # If only one leg is visible, use that one
    elif left_knee and left_hip:
        return left_leg_bent
    else:
        return right_leg_bent

def is_batting_stance(bat_bbox, pose_results, roi, frame_height, frame_width):
    """Check if the batter is in a proper batting stance.
    
    A proper batting stance requires:
    1. The bat is being held in the hands
    2. The bat is above the hands
    3. The hands are at shoulder level
    4. The knees are bent
    
    Args:
        bat_bbox: Bounding box of the bat
        pose_results: MediaPipe pose results
        roi: Region of interest containing the batter
        frame_height: Height of the full frame
        frame_width: Width of the full frame
        frame: The video frame to draw on
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
    
    # Get knee and hip landmarks
    left_knee = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
    left_hip = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    
    # Convert to full frame coordinates for bat-related checks
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
    
    # Draw circles at hand coordinates with larger radius and outline
    left_x, left_y = map(int, full_frame_left_hand)
    right_x, right_y = map(int, full_frame_right_hand)
    
    # Debug output
    print(f"\nHand Coordinates Debug:")
    print(f"Left hand: ({left_x}, {left_y})")
    print(f"Right hand: ({right_x}, {right_y})")
    if bat_bbox:
        bat_x1, bat_y1, bat_x2, bat_y2 = bat_bbox
        print(f"Bat box: ({bat_x1}, {bat_y1}) to ({bat_x2}, {bat_y2})")
    
    # Perform the checks using appropriate coordinate systems
    return (is_bat_in_hands(bat_bbox, full_frame_left_hand, full_frame_right_hand, frame_width, frame_height) and
            is_bat_above_hands(bat_bbox, full_frame_left_hand, full_frame_right_hand) and
            are_hands_at_shoulders(left_wrist, right_wrist, left_shoulder, right_shoulder) and
            are_knees_bent(left_knee, right_knee, left_hip, right_hip))

def detect_objects(frame, model, position_model):
    """Detect all objects in the frame using both YOLO and position models."""
    detections = {}
    
    # Use position model with tracking for players
    position_results = position_model.track(frame, persist=True, tracker="bytetrack.yaml")
    for r in position_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            class_name = r.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            
            if class_name not in detections:
                detections[class_name] = []
            detections[class_name].append((bbox, track_id, conf))
            
            label = f"{class_name} {conf:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            
            # Set text and box colors based on class
            if class_name == "Batter":
                text_color = (0, 0, 255)  # Red
                box_color = (0, 0, 255)   # Red
            elif class_name == "P":
                text_color = (255, 0, 255)  # Purple
                box_color = (255, 0, 255)   # Purple
            else:
                text_color = (255, 0, 0)  # Blue
                box_color = (255, 0, 0)   # Blue
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            print(f"{class_name} DETECTED! Confidence: {conf:.2f} ID:{track_id}")
    
    # Use YOLO model for bat detection
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            class_name = r.names[cls]
            
            if class_name == "baseball bat" and conf > 0.5:  # Only detect bats with high confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                
                # Print bat dimensions
                width = x2 - x1
                height = y2 - y1
                print(f"Bat dimensions: {width}x{height} pixels")
                
                if class_name not in detections:
                    detections[class_name] = []
                detections[class_name].append((bbox, track_id, conf))
                
                label = f"{class_name} {conf:.2f}"
                if track_id is not None:
                    label += f" ID:{track_id}"
                
                # Draw bat bounding box in green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"{class_name} DETECTED! Confidence: {conf:.2f} ID:{track_id}")
    
    return detections

def estimate_pose(frame, pose, mp_drawing, roi_x1, roi_y1, roi_x2, roi_y2):
    
    # Extract ROI and process pose
    roi_rgb = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2RGB)
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    pose_results = pose.process(roi_rgb)
    
    if pose_results.pose_landmarks:
        # Debug raw MediaPipe coordinates
        left_wrist = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        print(f"\nRaw MediaPipe coordinates:")
        print(f"Left wrist: ({left_wrist.x:.3f}, {left_wrist.y:.3f})")
        print(f"Right wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f})")
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            roi_frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )
    
    return roi_frame

def detect_batter_pose(video_path, save_output=False, playback_fps=15, start_frame=0):
    """Main function to detect batter poses in video.
    
    Args:
        video_path: Path to the video file
        save_output: Whether to save the output video
        playback_fps: Frame rate for playback
        model_name: Name of the YOLO model to use
        start_frame: Frame number to start processing from (0-based)
    """
    # Initialize models and video capture
    model, position_model, pose, mp_drawing = init_models()
    cap, out, cap_info = setup_video_capture(video_path, save_output)
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
        print(f"\nProcessing frame {frame_count}/{cap_info['total_frames']}")
        
        # Draw frame counter
        draw_frame_info(frame, frame_count, cap_info['total_frames'], cap_info['height'])
        
        # Process frame
        # Detect objects
        detections = detect_objects(frame, model, position_model)

            # Get batter bbox
        batter_bbox = None
        if "Batter" in detections:
            # Get the batter with highest confidence
            batters = detections["Batter"]
            batter_bbox, _, _ = max(batters, key=lambda x: x[2])

            # Process pose if we found a batter
        if batter_bbox:
            # Calculate ROI
            px1, py1, px2, py2 = batter_bbox
            padding = 20
            roi = (
                max(0, px1 - padding),
                max(0, py1 - padding),
                min(cap_info['width'], px2 + padding),
                min(cap_info['height'], py2 + padding)
            )
            """Process pose detection for the identified batter."""
        
        # Estimate pose
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        roi_frame = estimate_pose(frame, pose, mp_drawing, roi_x1, roi_y1, roi_x2, roi_y2)
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi_frame
        
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

def main():
    parser = argparse.ArgumentParser(description='Detect batter pose in video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--fps', type=int, default=30, help='Playback frame rate (default: 30)')
    parser.add_argument('--start-frame', type=int, default=0,
                      help='Frame number to start processing from (0-based, default: 0)')
    args = parser.parse_args()
    
    detect_batter_pose(args.video_path, args.save, args.fps, args.start_frame)

if __name__ == "__main__":
    main() 