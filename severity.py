def estimate_severity(box, frame_width, frame_height):
    """Calculate the severity of a bounding box based on its area relative to the frame.

    Args:
        box (tuple): (x1, y1, x2, y2) coordinates of the bounding box.
        frame_width (int): Width of the video frame.
        frame_height (int): Height of the video frame.

    Returns:
        str: 'Low', 'Medium', or 'High' based on the area ratio.
    """
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    ratio = box_area / frame_area
    if ratio < 0.01:
        return 'Low'
    elif ratio < 0.04:
        return 'Medium'
    else:
        return 'High'