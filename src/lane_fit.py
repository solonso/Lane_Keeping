import numpy as np


class LaneSearchParams:
    def __init__(
        self,
        num_windows=9,
        margin=80,
        minpix=40,
        min_pixels_for_fit=100,
        target_pixel_count=1000,
        residual_norm=25.0,
        detection_threshold=0.60,
        smooth_prior_weight=150.0,
        min_lane_width_px=150,
        max_lane_width_px=900,
        max_curvature_diff_ratio=3.0,
    ):
        self.num_windows = num_windows
        self.margin = margin
        self.minpix = minpix
        self.min_pixels_for_fit = min_pixels_for_fit
        self.target_pixel_count = target_pixel_count
        self.residual_norm = residual_norm
        self.detection_threshold = detection_threshold
        self.smooth_prior_weight = smooth_prior_weight
        self.min_lane_width_px = min_lane_width_px
        self.max_lane_width_px = max_lane_width_px
        self.max_curvature_diff_ratio = max_curvature_diff_ratio


class LaneLine:
    def __init__(self, coeffs, pixels, confidence, detected, curvature_m):
        self.coeffs = coeffs
        self.pixels = pixels
        self.confidence = confidence
        self.detected = detected
        self.curvature_m = curvature_m


class LaneDetection:
    def __init__(self, left, right, plot_y):
        self.left = left
        self.right = right
        self.plot_y = plot_y


def _initial_window_centers(
    histogram,
    image_height,
    prev_fit,
):
    midpoint = histogram.shape[0] // 2
    if prev_fit is not None:
        y_eval = image_height - 1
        x_est = np.polyval(prev_fit, y_eval)
        return int(np.clip(x_est, 0, len(histogram) - 1))
    left_half = histogram[:midpoint]
    if left_half.max() > 0:
        return int(np.argmax(left_half))
    return midpoint // 2


def _initial_right_center(histogram, image_height, prev_fit):
    midpoint = histogram.shape[0] // 2
    if prev_fit is not None:
        y_eval = image_height - 1
        x_est = np.polyval(prev_fit, y_eval)
        return int(np.clip(x_est, 0, len(histogram) - 1))
    right_half = histogram[midpoint:]
    if right_half.max() > 0:
        return int(np.argmax(right_half) + midpoint)
    return midpoint + midpoint // 2


def _compute_confidence(
    x,
    y,
    coeffs,
    params,
    prev_fit,
    debug=False,
):
    if coeffs is None or len(x) < params.min_pixels_for_fit:
        return 0.0
    count_score = min(len(x) / params.target_pixel_count, 1.0)
    fit_x = np.polyval(coeffs, y)
    residual = np.mean(np.abs(fit_x - x)) if len(x) > 0 else params.residual_norm
    residual_score = max(0.0, 1.0 - residual / params.residual_norm)
    if prev_fit is not None:
        y_eval = y.max() if y.size else 0
        prev_x = np.polyval(prev_fit, y_eval)
        curr_x = np.polyval(coeffs, y_eval)
        delta = abs(curr_x - prev_x)
        prior_score = max(0.0, 1.0 - delta / params.smooth_prior_weight)
    else:
        prior_score = 0.8
    confidence = 0.6 * count_score + 0.25 * residual_score + 0.15 * prior_score
    
    if debug:
        print(f"    Confidence debug: count={count_score:.3f}, residual={residual:.1f}, residual_score={residual_score:.3f}, prior={prior_score:.3f} -> conf={confidence:.3f}")
    
    return float(np.clip(confidence, 0.0, 1.0))


def _radius_of_curvature(
    coeffs,
    y_eval,
    ym_per_pix,
    xm_per_pix,
):
    if coeffs is None:
        return None
    fit_cr = np.array(
        [
            coeffs[0] * xm_per_pix / (ym_per_pix ** 2),
            coeffs[1] * xm_per_pix / ym_per_pix,
            coeffs[2],
        ]
    )
    denominator = abs(2 * fit_cr[0])
    if denominator < 1e-6:
        return None
    return ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / denominator


def _validate_lane_pair(left_fit, right_fit, image_width, image_height, params, debug=False):
    """Check if detected lane pair is reasonable."""
    if left_fit is None or right_fit is None:
        if debug:
            print(f"  Validation: One or both fits are None (left={left_fit is not None}, right={right_fit is not None})")
        return True  # Individual lanes validated separately
    
    # Check lane width at multiple points (bottom and middle only - top distorts on curves)
    y_samples = np.array([image_height - 1, image_height * 2 // 3, image_height // 3])
    widths = []
    
    for y in y_samples:
        left_x = np.polyval(left_fit, y)
        right_x = np.polyval(right_fit, y)
        width = right_x - left_x
        
        if debug:
            print(f"  Validation: y={y:.0f}, left_x={left_x:.1f}, right_x={right_x:.1f}, width={width:.1f}")
        
        # Lane width must be positive and in reasonable range
        if width < params.min_lane_width_px or width > params.max_lane_width_px:
            if debug:
                print(f"  Validation FAILED: Width {width:.1f} outside range [{params.min_lane_width_px}, {params.max_lane_width_px}]")
            return False
        widths.append(width)
    
    # Check if lanes are roughly parallel (width variation < 120% for curves)
    width_variation = (max(widths) - min(widths)) / np.mean(widths)
    if debug:
        print(f"  Validation: Width variation={width_variation:.2f} (max allowed=1.2)")
    if width_variation > 1.2:
        if debug:
            print(f"  Validation FAILED: Width variation too high")
        return False
    
    if debug:
        print(f"  Validation PASSED")
    return True


def sliding_window_fit(
    binary_warped,
    params,
    prev_left_fit=None,
    prev_right_fit=None,
    ym_per_pix=30 / 720,
    xm_per_pix=3.7 / 700,
    debug_validation=False,
):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    leftx_current = _initial_window_centers(histogram, binary_warped.shape[0], prev_left_fit)
    rightx_current = _initial_right_center(histogram, binary_warped.shape[0], prev_right_fit)

    window_height = int(binary_warped.shape[0] / params.num_windows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    for window in range(params.num_windows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - params.margin
        win_xleft_high = leftx_current + params.margin
        win_xright_low = rightx_current - params.margin
        win_xright_high = rightx_current + params.margin

        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > params.minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > params.minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds_concat = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds_concat = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    leftx = nonzerox[left_lane_inds_concat]
    lefty = nonzeroy[left_lane_inds_concat]
    rightx = nonzerox[right_lane_inds_concat]
    righty = nonzeroy[right_lane_inds_concat]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) >= params.min_pixels_for_fit else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) >= params.min_pixels_for_fit else None

    # Baby step fix: For very low coverage with unstable top, use linear fit (more stable extrapolation)
    height = binary_warped.shape[0]
    if left_fit is not None and len(lefty) > 0:
        pixel_coverage = (lefty.max() - lefty.min()) / height
        # Only for low coverage (<40%) where quadratic extrapolation can be unreliable
        if pixel_coverage < 0.40:
            top_x = np.polyval(left_fit, 0)
            x_mean = leftx.mean()
            x_std = leftx.std()
            # If top point is very far from pixels, linear fit is more stable
            if abs(top_x - x_mean) > max(150, 3 * x_std):
                # Use linear fit instead - more stable for sparse data
                left_fit = np.polyfit(lefty, leftx, 1)
                # Convert to quadratic form [0, slope, intercept] for compatibility
                left_fit = np.array([0.0, left_fit[0], left_fit[1]])
    
    if right_fit is not None and len(righty) > 0:
        pixel_coverage = (righty.max() - righty.min()) / height
        if pixel_coverage < 0.40:
            top_x = np.polyval(right_fit, 0)
            x_mean = rightx.mean()
            x_std = rightx.std()
            if abs(top_x - x_mean) > max(150, 3 * x_std):
                right_fit = np.polyfit(righty, rightx, 1)
                right_fit = np.array([0.0, right_fit[0], right_fit[1]])

    plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_curvature = (
        _radius_of_curvature(left_fit, np.max(plot_y), ym_per_pix, xm_per_pix)
        if left_fit is not None
        else None
    )
    right_curvature = (
        _radius_of_curvature(right_fit, np.max(plot_y), ym_per_pix, xm_per_pix)
        if right_fit is not None
        else None
    )

    # Validate lane pair geometry
    lanes_valid = _validate_lane_pair(left_fit, right_fit, binary_warped.shape[1], binary_warped.shape[0], params, debug=debug_validation)
    
    if not lanes_valid:
        # Reject both lanes if geometry is invalid
        left_fit = None
        right_fit = None
        left_conf = 0.0
        right_conf = 0.0
    else:
        left_conf = _compute_confidence(leftx, lefty, left_fit, params, prev_left_fit, debug=debug_validation)
        right_conf = _compute_confidence(rightx, righty, right_fit, params, prev_right_fit, debug=debug_validation)

    left_detected = left_conf > params.detection_threshold
    right_detected = right_conf > params.detection_threshold

    left_line = LaneLine(
        coeffs=left_fit,
        pixels=(leftx, lefty),
        confidence=left_conf,
        detected=left_detected,
        curvature_m=left_curvature,
    )
    right_line = LaneLine(
        coeffs=right_fit,
        pixels=(rightx, righty),
        confidence=right_conf,
        detected=right_detected,
        curvature_m=right_curvature,
    )

    return LaneDetection(left=left_line, right=right_line, plot_y=plot_y)


def lane_center_offset_px(
    detection,
    image_width,
    image_height,
):
    if detection.left.coeffs is None or detection.right.coeffs is None:
        return None, None
    y_eval = image_height - 1
    left_x = np.polyval(detection.left.coeffs, y_eval)
    right_x = np.polyval(detection.right.coeffs, y_eval)
    lane_width_px = right_x - left_x
    if lane_width_px <= 0:
        return None, None
    lane_center = (left_x + right_x) / 2.0
    vehicle_center = image_width / 2.0
    return float(vehicle_center - lane_center), float(lane_width_px)
