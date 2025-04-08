import pytest
from detect_batter import roi_to_full_frame

def test_roi_to_full_frame():
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

    print("All tests passed!")

if __name__ == "__main__":
    test_roi_to_full_frame() 