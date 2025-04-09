import pytest
from detect_batter import roi_to_full_frame

def test_roi_to_full_frame():
    """Test roi_to_full_frame function with various cases."""
    # Test case 1: Simple conversion with no scaling needed
    # ROI is 100x100 at position (50,50) in full frame
    # Point at (0.5, 0.5) in normalized ROI should map to (100, 100) in full frame
    result = roi_to_full_frame(0.5, 0.5, 50, 50, 100, 100)
    assert result == (100, 100), f"Expected (100, 100), got {result}"

    # Test case 2: Point at ROI origin
    # Point at (0, 0) in normalized ROI should map to ROI's top-left corner
    result = roi_to_full_frame(0, 0, 50, 50, 100, 100)
    assert result == (50, 50), f"Expected (50, 50), got {result}"

    # Test case 3: Point at ROI bottom-right
    # Point at (1, 1) in normalized ROI should map to ROI's bottom-right corner
    result = roi_to_full_frame(1, 1, 50, 50, 100, 100)
    assert result == (150, 150), f"Expected (150, 150), got {result}"

    # Test case 4: Different ROI dimensions
    # ROI is 200x50 at position (0,0)
    # Point at (0.5, 0.5) should map to (100, 25)
    result = roi_to_full_frame(0.5, 0.5, 0, 0, 200, 50)
    assert result == (100, 25), f"Expected (100, 25), got {result}"

def test_is_bat_in_hands():
    """Test is_bat_in_hands function with various cases."""
    from detect_batter import is_bat_in_hands

    # Test frame dimensions (using 1000x1000 as a reasonable test size)
    frame_width = 1000
    frame_height = 1000
    max_hand_distance = frame_width * 0.07  # 7% of frame width
    max_vertical_distance = frame_height * 0.09  # 9% of frame height

    # Test case 1: Bat being held correctly
    # Bat bottom at y=100, hands at y=90 and y=95, close together horizontally
    bat_bbox = (50, 0, 70, 100)  # x1, y1, x2, y2
    left_hand = (55, 90)  # x, y
    right_hand = (65, 95)  # x, y
    assert is_bat_in_hands(bat_bbox, left_hand, right_hand, frame_width, frame_height), "Should detect bat being held"

    # Test case 2: Hands too far apart horizontally
    # Same y positions but hands more than 7% of frame width apart
    bat_bbox = (50, 0, 70, 100)
    left_hand = (0, 90)
    right_hand = (frame_width * 0.08, 95)  # 8% of frame width apart
    assert not is_bat_in_hands(bat_bbox, left_hand, right_hand, frame_width, frame_height), "Should not detect bat being held when hands too far apart"

    # Test case 3: Hands too far vertically from bat
    # Hands more than 9% of frame height away from bat bottom
    bat_bbox = (50, 0, 70, 100)
    left_hand = (55, 200)  # 100px away from bat bottom (10% of frame height)
    right_hand = (65, 200)
    assert not is_bat_in_hands(bat_bbox, left_hand, right_hand, frame_width, frame_height), "Should not detect bat being held when hands too far vertically"

def test_are_hands_at_shoulders():
    """Test are_hands_at_shoulders function with various cases."""
    from detect_batter import are_hands_at_shoulders

    # Test frame height (using 1000 pixels as a reasonable test size)
    frame_height = 1000
    max_distance = frame_height * 0.02  # 2% of frame height

    # Test case 1: Hands at shoulder level
    # Shoulders at y=500, hands within 2% of frame height
    left_hand = (50, 480)  # x, y (20 pixels above shoulders)
    right_hand = (150, 520)  # x, y (20 pixels below shoulders)
    left_shoulder = (40, 500)  # x, y
    right_shoulder = (160, 500)  # x, y
    assert are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should detect hands at shoulder level"

    # Test case 2: Hands too far above shoulders
    # Shoulders at y=500, hands more than 2% above
    left_hand = (50, 300)  # x, y (200 pixels above shoulders)
    right_hand = (150, 310)  # x, y (190 pixels above shoulders)
    left_shoulder = (40, 500)
    right_shoulder = (160, 500)
    assert not are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should not detect hands at shoulder level when too high"

    # Test case 3: Hands too far below shoulders
    # Shoulders at y=500, hands more than 2% below
    left_hand = (50, 700)  # x, y (200 pixels below shoulders)
    right_hand = (150, 710)  # x, y (210 pixels below shoulders)
    left_shoulder = (40, 500)
    right_shoulder = (160, 500)
    assert not are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should not detect hands at shoulder level when too low"

    # Test case 4: Missing landmarks
    # Only left hand and left shoulder present
    left_hand = (50, 495)
    right_hand = None
    left_shoulder = (40, 500)
    right_shoulder = None
    assert not are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should not detect hands at shoulder level with missing landmarks"

def test_is_bat_above_hands():
    """Test is_bat_above_hands function with various cases."""
    from detect_batter import is_bat_above_hands

    # Test case 1: Bat above both hands
    # Bat center at y=50, hands at y=100 and y=120
    bat_bbox = (50, 0, 70, 100)  # x1, y1, x2, y2 (center at y=50)
    left_hand = (55, 100)  # x, y
    right_hand = (65, 120)  # x, y
    assert is_bat_above_hands(bat_bbox, left_hand, right_hand), "Should detect bat above both hands"

    # Test case 2: Bat above one hand but not the other
    # Bat center at y=50, left hand at y=100, right hand at y=40
    bat_bbox = (50, 0, 70, 100)  # x1, y1, x2, y2 (center at y=50)
    left_hand = (55, 100)  # x, y
    right_hand = (65, 40)  # x, y
    assert not is_bat_above_hands(bat_bbox, left_hand, right_hand), "Should not detect bat above both hands when only above one"

    # Test case 3: Bat below both hands
    # Bat center at y=150, hands at y=100 and y=120
    bat_bbox = (50, 100, 70, 200)  # x1, y1, x2, y2 (center at y=150)
    left_hand = (55, 100)  # x, y
    right_hand = (65, 120)  # x, y
    assert not is_bat_above_hands(bat_bbox, left_hand, right_hand), "Should not detect bat above both hands when below both"

    # Test case 4: Missing landmarks
    # Only left hand present
    bat_bbox = (50, 0, 70, 100)
    left_hand = (55, 100)
    right_hand = None
    assert not is_bat_above_hands(bat_bbox, left_hand, right_hand), "Should not detect bat above both hands with missing hand"

    # Test case 5: Missing bat
    # No bat bbox
    bat_bbox = None
    left_hand = (55, 100)
    right_hand = (65, 120)
    assert not is_bat_above_hands(bat_bbox, left_hand, right_hand), "Should not detect bat above both hands with missing bat"

def test_are_knees_bent():
    """Test are_knees_bent function with various cases."""
    from detect_batter import are_knees_bent
    from mediapipe.framework.formats import landmark_pb2

    def create_landmark(x, y):
        landmark = landmark_pb2.NormalizedLandmark()
        landmark.x = x
        landmark.y = y
        return landmark

    # Test case 1: No legs visible - should return True
    assert are_knees_bent(None, None, None, None), "Should return True when no legs are visible"

    # Test case 2: Only left leg visible and bent (12 degrees) - should return True
    left_knee = create_landmark(105.7, 121.4)  # 12 degree angle (dx=5.7, dy=21.4, dx/dy≈0.213)
    left_hip = create_landmark(100, 100)
    assert are_knees_bent(left_knee, None, left_hip, None), "Should return True when only left leg is visible and bent"

    # Test case 3: Only left leg visible and straight (5 degrees) - should return False
    left_knee = create_landmark(100, 108.7)  # 5 degree angle (dx=0, dy=8.7, dx/dy≈0)
    left_hip = create_landmark(100, 100)
    assert not are_knees_bent(left_knee, None, left_hip, None), "Should return False when only left leg is visible and straight"

    # Test case 4: Only right leg visible and bent (12 degrees) - should return True
    right_knee = create_landmark(294.3, 121.4)  # 12 degree angle (dx=5.7, dy=21.4, dx/dy≈0.213)
    right_hip = create_landmark(300, 100)
    assert are_knees_bent(None, right_knee, None, right_hip), "Should return True when only right leg is visible and bent"

    # Test case 5: Only right leg visible and straight (5 degrees) - should return False
    right_knee = create_landmark(300, 108.7)  # 5 degree angle (dx=0, dy=8.7, dx/dy≈0)
    right_hip = create_landmark(300, 100)
    assert not are_knees_bent(None, right_knee, None, right_hip), "Should return False when only right leg is visible and straight"

    # Test case 6: Both legs visible, left straight (5 degrees), right bent (12 degrees) - should return False
    left_knee = create_landmark(100, 108.7)  # 5 degree angle
    right_knee = create_landmark(294.3, 121.4)  # 12 degree angle
    left_hip = create_landmark(100, 100)
    right_hip = create_landmark(300, 100)
    assert not are_knees_bent(left_knee, right_knee, left_hip, right_hip), "Should return False when left leg is straight and right leg is bent"

    # Test case 7: Both legs visible, right straight (5 degrees), left bent (12 degrees) - should return False
    left_knee = create_landmark(105.7, 121.4)  # 12 degree angle
    right_knee = create_landmark(300, 108.7)  # 5 degree angle
    left_hip = create_landmark(100, 100)
    right_hip = create_landmark(300, 100)
    assert not are_knees_bent(left_knee, right_knee, left_hip, right_hip), "Should return False when right leg is straight and left leg is bent"

    # Test case 8: Both legs visible and both bent (12 degrees) - should return True
    left_knee = create_landmark(105.7, 121.4)  # 12 degree angle
    right_knee = create_landmark(294.3, 121.4)  # 12 degree angle
    left_hip = create_landmark(100, 100)
    right_hip = create_landmark(300, 100)
    assert are_knees_bent(left_knee, right_knee, left_hip, right_hip), "Should return True when both legs are bent"

if __name__ == "__main__":
    test_roi_to_full_frame()
    test_is_bat_in_hands()
    test_are_hands_at_shoulders()
    test_is_bat_above_hands()
    test_are_knees_bent()
    print("All tests passed!") 