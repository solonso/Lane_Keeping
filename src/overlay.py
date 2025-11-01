import cv2
import numpy as np

from .lane_fit import LaneDetection


class OverlayConfig:
    def __init__(
        self,
        roi_top_ratio,
        left_color=(40, 200, 255),
        right_color=(255, 160, 0),
        fallback_color=(180, 180, 180),
        lane_fill_color=(70, 170, 90),
        thickness=10,
    ):
        self.roi_top_ratio = roi_top_ratio
        self.left_color = left_color
        self.right_color = right_color
        self.fallback_color = fallback_color
        self.lane_fill_color = lane_fill_color
        self.thickness = thickness


def _poly_points(coeffs, plot_y):
    fit_x = np.polyval(coeffs, plot_y)
    points = np.vstack((fit_x, plot_y)).T
    return np.int32(points)


def _draw_line(
    image,
    points,
    color,
    thickness,
    dashed=False,
    dash_length=20,
):
    if len(points) < 2:
        return
    step = dash_length
    for i in range(len(points) - 1):
        if dashed and (i // step) % 2 == 1:
            continue
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        cv2.line(image, pt1, pt2, color, thickness)


def draw_lane_overlay(
    frame,
    detection: LaneDetection,
    inverse_perspective,
    warp_shape,
    config: OverlayConfig,
    left_conf,
    right_conf,
    left_detected,
    right_detected,
    lateral_offset_m,
    detection_stats=None,
):
    # With ROI mask approach, we work with full frame dimensions
    overlay_warp = np.zeros((warp_shape[0], warp_shape[1], 3), dtype=np.uint8)

    left_points = (
        _poly_points(detection.left.coeffs, detection.plot_y)
        if detection.left.coeffs is not None
        else None
    )
    right_points = (
        _poly_points(detection.right.coeffs, detection.plot_y)
        if detection.right.coeffs is not None
        else None
    )

    if left_points is not None and right_points is not None:
        lane_polygon = np.vstack(
            (left_points, np.flipud(right_points))
        ).reshape((-1, 1, 2))
        cv2.fillPoly(overlay_warp, [lane_polygon], config.lane_fill_color)

    if left_points is not None:
        _draw_line(
            overlay_warp,
            left_points,
            config.left_color if left_detected else config.fallback_color,
            config.thickness,
            dashed=not left_detected,
        )
    if right_points is not None:
        _draw_line(
            overlay_warp,
            right_points,
            config.right_color if right_detected else config.fallback_color,
            config.thickness,
            dashed=not right_detected,
        )

    overlay_full = cv2.warpPerspective(overlay_warp, inverse_perspective, (frame.shape[1], frame.shape[0]))
    output = frame.copy()
    overlay_canvas = overlay_full
    output = cv2.addWeighted(output, 1.0, overlay_canvas, 0.45, 0)

    panel_top = (20, 20)
    panel_bottom = (360, 165)
    cv2.rectangle(output, panel_top, panel_bottom, (15, 15, 15), -1)
    cv2.rectangle(output, panel_top, panel_bottom, (75, 75, 75), 2)

    cv2.putText(output, "LANE CONFIDENCE", (38, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210, 210, 210), 2, cv2.LINE_AA)

    def draw_status(x, y, label, detected, confidence, color):
        status_color = color if detected else config.fallback_color
        cv2.circle(output, (x, y - 6), 7, status_color, -1)
        cv2.putText(output, f"{label.upper()} CONF", (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (225, 225, 225), 2, cv2.LINE_AA)
        if detected:
            cv2.putText(output, f"{confidence:.2f}", (x + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            cv2.putText(output, "--", (x + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2, cv2.LINE_AA)

    draw_status(38, 86, "Left", left_detected, left_conf, config.left_color)
    draw_status(38, 118, "Right", right_detected, right_conf, config.right_color)

    if lateral_offset_m is not None:
        offset_text = f"Offset {lateral_offset_m:+.2f} m"
    else:
        offset_text = "Offset --"
    cv2.putText(output, offset_text, (38, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (210, 210, 210), 2, cv2.LINE_AA)

    # Right-side dashboard for detection stats
    if detection_stats is not None:
        frame_width = frame.shape[1]
        right_panel_left = frame_width - 360
        right_panel_top = (right_panel_left, 20)
        right_panel_bottom = (frame_width - 20, 165)
        
        cv2.rectangle(output, right_panel_top, right_panel_bottom, (15, 15, 15), -1)
        cv2.rectangle(output, right_panel_top, right_panel_bottom, (75, 75, 75), 2)
        
        cv2.putText(output, "DETECTION STATS", (right_panel_left + 22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210, 210, 210), 2, cv2.LINE_AA)
        
        # Detection percentage
        detection_pct = detection_stats.get('detection_rate', 0.0)
        cv2.putText(output, f"Detection: {detection_pct:.1f}%", (right_panel_left + 22, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (225, 225, 225), 2, cv2.LINE_AA)
        
        # Flickers per 10s
        flickers = detection_stats.get('flickers_per_10s', 0.0)
        flicker_color = (0, 255, 0) if flickers < 1.0 else (0, 0, 255)
        cv2.putText(output, f"Flickers/10s: {flickers:.2f}", (right_panel_left + 22, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.64, flicker_color, 2, cv2.LINE_AA)
        
        # Frame count
        frame_count = detection_stats.get('frame_count', 0)
        cv2.putText(output, f"Frames: {frame_count}", (right_panel_left + 22, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (210, 210, 210), 2, cv2.LINE_AA)

    return output
