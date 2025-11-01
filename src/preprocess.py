import cv2
import numpy as np


class PreprocessParams:
    def __init__(
        self,
        roi_top_ratio=0.68,
        roi_bottom_margin=0.0,
        s_thresh=(120, 255),
        v_thresh=(200, 255),
        l_thresh=(200, 255),
        sobel_thresh=(30, 255),
        sobel_kernel=3,
        morph_kernel=3,
        morph_iterations=1,
    ):
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_margin = roi_bottom_margin
        self.s_thresh = s_thresh
        self.v_thresh = v_thresh
        self.l_thresh = l_thresh
        self.sobel_thresh = sobel_thresh
        self.sobel_kernel = sobel_kernel
        self.morph_kernel = morph_kernel
        self.morph_iterations = morph_iterations
        
def create_roi_mask(frame_shape, params):
    height, width = frame_shape[:2]
    top_y = int(height * params.roi_top_ratio)
    taper = int(width * 0.18)
    bottom_margin = int(width * params.roi_bottom_margin)

    vertices = np.array(
        [[
            (int(width * 0.5 - taper), top_y),
            (int(width * 0.5 + taper), top_y),
            (width - bottom_margin, height),
            (bottom_margin, height),
        ]],
        dtype=np.int32,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 1)
    return mask


def threshold_lane_pixels(frame, params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    v_channel = hsv[:, :, 2]
    h_channel = hsv[:, :, 0]

    # Yellow lane detection (H: 15-35, S: 80-255, V: 120-255)
    yellow_mask = np.zeros_like(gray, dtype=np.uint8)
    yellow_mask[
        (h_channel >= 15) & (h_channel <= 35) &
        (s_channel >= 80) & 
        (v_channel >= 120)
    ] = 1

    # White lane detection (high V, low S)
    white_mask = np.zeros_like(gray, dtype=np.uint8)
    white_mask[
        (s_channel <= 40) & 
        (v_channel >= 180)
    ] = 1

    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=params.sobel_kernel)
    abs_sobel_x = np.absolute(sobel_x)
    if abs_sobel_x.max() > 0:
        sobel_scaled = np.uint8(255 * abs_sobel_x / abs_sobel_x.max())
    else:
        sobel_scaled = np.zeros_like(l_channel, dtype=np.uint8)

    gray_binary = np.zeros_like(gray, dtype=np.uint8)
    gray_binary[(gray >= params.l_thresh[0]) & (gray <= params.l_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel, dtype=np.uint8)
    s_binary[(s_channel >= params.s_thresh[0]) & (s_channel <= params.s_thresh[1])] = 1

    v_binary = np.zeros_like(v_channel, dtype=np.uint8)
    v_binary[(v_channel >= params.v_thresh[0]) & (v_channel <= params.v_thresh[1])] = 1

    sobel_binary = np.zeros_like(sobel_scaled, dtype=np.uint8)
    sobel_binary[(sobel_scaled >= params.sobel_thresh[0]) & (sobel_scaled <= params.sobel_thresh[1])] = 1

    # Prioritize color-specific detection
    color_binary = np.zeros_like(s_channel, dtype=np.uint8)
    color_binary[(yellow_mask == 1) | (white_mask == 1)] = 1
    
    # Fallback to general thresholds if color detection is weak
    general_binary = np.zeros_like(s_channel, dtype=np.uint8)
    general_binary[(s_binary == 1) & (v_binary == 1) & (gray_binary == 1)] = 1

    combined = np.zeros_like(color_binary, dtype=np.uint8)
    combined[(color_binary == 1) | (sobel_binary == 1) | (general_binary == 1)] = 1

    roi_mask = create_roi_mask(frame.shape, params)
    final_binary = np.zeros_like(combined, dtype=np.uint8)
    final_binary[(roi_mask == 1) & (combined == 1)] = 1

    # Clean up noise: close gaps, remove small components
    if params.morph_kernel > 1 and params.morph_iterations > 0:
        kernel = np.ones((params.morph_kernel, params.morph_kernel), np.uint8)
        # Close gaps in sparse detections
        closed = cv2.morphologyEx(final_binary * 255, cv2.MORPH_CLOSE, kernel, iterations=params.morph_iterations)
        # Remove small noise components
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        final_binary = (opened > 0).astype(np.uint8)
        
        # Remove very small connected components (< 50 pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_binary, connectivity=8)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] < 50:
                final_binary[labels == i] = 0

    return final_binary
