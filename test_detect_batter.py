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

    # Test case 1: Bat being held correctly
    # Bat bottom at y=100, hands at y=90 and y=95, close together horizontally
    bat_bbox = (50, 0, 70, 100)  # x1, y1, x2, y2
    left_hand = (55, 90)  # x, y
    right_hand = (65, 95)  # x, y
    assert is_bat_in_hands(bat_bbox, left_hand, right_hand), "Should detect bat being held"

    # Test case 2: Hands too far apart horizontally
    # Same y positions but hands 100 pixels apart
    bat_bbox = (50, 0, 70, 100)
    left_hand = (0, 90)
    right_hand = (100, 95)
    assert not is_bat_in_hands(bat_bbox, left_hand, right_hand), "Should not detect bat being held when hands too far apart"

    # Test case 3: Hands too far from bat bottom vertically
    # Hands 100 pixels above bat bottom
    bat_bbox = (50, 0, 70, 100)
    left_hand = (55, 0)
    right_hand = (65, 0)
    assert not is_bat_in_hands(bat_bbox, left_hand, right_hand), "Should not detect bat being held when hands too far from bottom"

    # Test case 4: Hands at bat bottom but one hand missing
    # Only left hand present
    bat_bbox = (50, 0, 70, 100)
    left_hand = (55, 95)
    right_hand = None
    assert not is_bat_in_hands(bat_bbox, left_hand, right_hand), "Should not detect bat being held with missing hand"

    # Test case 5: Hands at bat bottom but bat missing
    # No bat bbox
    bat_bbox = None
    left_hand = (55, 95)
    right_hand = (65, 95)
    assert not is_bat_in_hands(bat_bbox, left_hand, right_hand), "Should not detect bat being held with missing bat"

def test_are_hands_at_shoulders():
    """Test are_hands_at_shoulders function with various cases."""
    from detect_batter import are_hands_at_shoulders

    # Test frame height (using 1000 pixels as a reasonable test size)
    frame_height = 1000
    max_distance = frame_height * 0.05  # 5% of frame height

    # Test case 1: Hands at shoulder level
    # Shoulders at y=500, hands within 5% of frame height
    left_hand = (50, 480)  # x, y (20 pixels above shoulders)
    right_hand = (150, 520)  # x, y (20 pixels below shoulders)
    left_shoulder = (40, 500)  # x, y
    right_shoulder = (160, 500)  # x, y
    assert are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should detect hands at shoulder level"

    # Test case 2: Hands too far above shoulders
    # Shoulders at y=500, hands more than 5% above
    left_hand = (50, 400)  # x, y (100 pixels above shoulders)
    right_hand = (150, 410)  # x, y (90 pixels above shoulders)
    left_shoulder = (40, 500)
    right_shoulder = (160, 500)
    assert not are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should not detect hands at shoulder level when too high"

    # Test case 3: Hands too far below shoulders
    # Shoulders at y=500, hands more than 5% below
    left_hand = (50, 600)  # x, y (100 pixels below shoulders)
    right_hand = (150, 610)  # x, y (110 pixels below shoulders)
    left_shoulder = (40, 500)
    right_shoulder = (160, 500)
    assert not are_hands_at_shoulders(left_hand, right_hand, left_shoulder, right_shoulder, frame_height), "Should not detect hands at shoulder level when too low"

    # Test case 4: Missing landmarks
    # Only left hand and left shoulder present
    left_hand = (50, 480)
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

if __name__ == "__main__":
    test_roi_to_full_frame()
    test_is_bat_in_hands()
    test_are_hands_at_shoulders()
    test_is_bat_above_hands()
    print("All tests passed!") 