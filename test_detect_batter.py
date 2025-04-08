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

if __name__ == "__main__":
    test_roi_to_full_frame()
    test_is_bat_in_hands()
    print("All tests passed!") 