import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .csv_writer import CSVLogger
from .lane_fit import sliding_window_fit, lane_center_offset_px, LaneSearchParams, LaneDetection
from .overlay import OverlayConfig, draw_lane_overlay
from .preprocess import PreprocessParams, threshold_lane_pixels, create_roi_mask
from .temporal import LaneTemporalFilter, TemporalParams
from .warp import WarpConfig, compute_warp_matrices, warp_image


def save_debug_images(frame, frame_id, binary_mask, warped_binary, detection, output_dir, preprocess_params):
    """Save debug visualization images for a single frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Show ROI mask
    roi_mask = create_roi_mask(frame.shape, preprocess_params)
    roi_visual = frame.copy()
    roi_visual[roi_mask == 0] = roi_visual[roi_mask == 0] // 3
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_01_roi.jpg"), roi_visual)
    
    # Step 2: Color channels
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_02a_L_channel.jpg"), hls[:,:,1])
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_02b_S_channel.jpg"), hls[:,:,2])
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_02c_V_channel.jpg"), hsv[:,:,2])
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_02d_gray.jpg"), gray)
    
    # Step 3: Thresholded masks
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_03_binary_mask.jpg"), binary_mask * 255)
    
    # Step 4: Warped binary
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_04_warped_binary.jpg"), warped_binary * 255)
    
    # Step 5: Histogram
    histogram = np.sum(warped_binary[warped_binary.shape[0] // 2:, :], axis=0)
    hist_plot = np.zeros((200, len(histogram), 3), dtype=np.uint8)
    if histogram.max() > 0:
        histogram_normalized = (histogram / histogram.max() * 180).astype(int)
        for i, val in enumerate(histogram_normalized):
            cv2.line(hist_plot, (i, 200), (i, 200 - val), (0, 255, 0), 1)
    midpoint = len(histogram) // 2
    cv2.line(hist_plot, (midpoint, 0), (midpoint, 200), (0, 0, 255), 2)
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_05_histogram.jpg"), hist_plot)
    
    # Step 6: Detected lanes
    warped_colored = np.dstack((warped_binary, warped_binary, warped_binary)) * 255
    plot_y = detection.plot_y
    
    if detection.left.coeffs is not None:
        left_fitx = np.polyval(detection.left.coeffs, plot_y)
        left_points = np.array([np.transpose(np.vstack([left_fitx, plot_y]))], dtype=np.int32)
        cv2.polylines(warped_colored, left_points, False, (255, 255, 0), 5)
        
    if detection.right.coeffs is not None:
        right_fitx = np.polyval(detection.right.coeffs, plot_y)
        right_points = np.array([np.transpose(np.vstack([right_fitx, plot_y]))], dtype=np.int32)
        cv2.polylines(warped_colored, right_points, False, (0, 160, 255), 5)
    
    cv2.imwrite(str(output_dir / f"frame_{frame_id:04d}_06_detected_lanes.jpg"), warped_colored)


def parse_args():
    parser = argparse.ArgumentParser(description="Classical CV lane keep assist baseline.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output_data", help="Annotated video output path.")
    parser.add_argument("--csv", help="Per-frame CSV output path.")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames.")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N frames.")
    parser.add_argument("--show", action="store_true", help="Display frames during processing.")
    return parser.parse_args()


def process_video(
    input_path,
    output_path,
    csv_path,
    args,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer at {output_path}")

    preprocess_params = PreprocessParams(roi_bottom_margin=0.0)
    overlay_config = OverlayConfig(roi_top_ratio=preprocess_params.roi_top_ratio)
    search_params = LaneSearchParams()
    temporal_filter = LaneTemporalFilter(TemporalParams())
    warp_config = WarpConfig()

    perspective_matrix: Optional[np.ndarray] = None
    inverse_matrix: Optional[np.ndarray] = None
    prev_left_fit = None
    prev_right_fit = None

    frame_id = -1
    processed_frames = 0

    lane_width_m = 3.7
    
    # Detection stats tracking
    detection_history = []
    flicker_count = 0

    with CSVLogger(csv_path) as logger:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id < args.skip:
                continue
            if args.max_frames and processed_frames >= args.max_frames:
                break
            
            processed_frames += 1

            binary_mask = threshold_lane_pixels(frame, preprocess_params)
            mask_uint8 = (binary_mask * 255).astype(np.uint8)

            if perspective_matrix is None or inverse_matrix is None:
                # Use ROI region dimensions for perspective transform
                perspective_matrix, inverse_matrix = compute_warp_matrices(mask_uint8.shape, warp_config)

            warped = warp_image(mask_uint8, perspective_matrix)
            warped_binary = (warped > 0).astype(np.uint8)

            detection: LaneDetection = sliding_window_fit(
                warped_binary,
                params=search_params,
                prev_left_fit=prev_left_fit,
                prev_right_fit=prev_right_fit,
            )

            detection = temporal_filter.apply(detection)
            prev_left_fit = temporal_filter.prev_left_fit
            prev_right_fit = temporal_filter.prev_right_fit

            offset_px, lane_width_px = lane_center_offset_px(
                detection,
                warped_binary.shape[1],
                warped_binary.shape[0],
            )
            if offset_px is not None and lane_width_px is not None and lane_width_px > 0:
                lateral_offset_m = (offset_px / lane_width_px) * lane_width_m
            else:
                lateral_offset_m = None

            # Track detection stats
            detection_history.append((detection.left.detected, detection.right.detected))
            
            # Count flickers (simple check for changes)
            if len(detection_history) >= 3:
                prev_left, prev_right = detection_history[-3][0], detection_history[-3][1]
                curr_left, curr_right = detection_history[-2][0], detection_history[-2][1]
                next_left, next_right = detection_history[-1][0], detection_history[-1][1]
                
                # Check for 1->0->1 or 0->1->0 patterns
                if (prev_left == 1 and curr_left == 0 and next_left == 1) or (prev_left == 0 and curr_left == 1 and next_left == 0):
                    flicker_count += 1
                if (prev_right == 1 and curr_right == 0 and next_right == 1) or (prev_right == 0 and curr_right == 1 and next_right == 0):
                    flicker_count += 1
            
            # Calculate current stats
            total_detections = sum(left + right for left, right in detection_history)
            total_possible = len(detection_history) * 2
            detection_rate = (total_detections / total_possible * 100) if total_possible > 0 else 0.0
            
            fps = 25  # Assume 25 fps
            frames_per_10s = fps * 10
            flickers_per_10s = (flicker_count * frames_per_10s / len(detection_history)) if len(detection_history) > 0 else 0.0
            
            detection_stats = {
                'detection_rate': detection_rate,
                'flickers_per_10s': flickers_per_10s,
                'frame_count': len(detection_history)
            }

            annotated = draw_lane_overlay(
                frame,
                detection,
                inverse_matrix,
                warped_binary.shape,
                overlay_config,
                detection.left.confidence,
                detection.right.confidence,
                detection.left.detected,
                detection.right.detected,
                lateral_offset_m,
                detection_stats,
            )

            writer.write(annotated)

            logger.write(
                frame_id=frame_id,
                left_detected=detection.left.detected,
                right_detected=detection.right.detected,
                left_conf=detection.left.confidence,
                right_conf=detection.right.confidence,
                lateral_offset_m=lateral_offset_m,
                left_curvature_m=detection.left.curvature_m,
                right_curvature_m=detection.right.curvature_m,
            )

            if args.show:
                cv2.imshow("annotated", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        # Save debug images from the first processed frame
        if processed_frames > 0:
            print("\nSaving debug visualization images...")
            debug_dir = csv_path.parent / f"{input_path.stem}_debug_frames"
            # Reload first frame to save debug images
            debug_cap = cv2.VideoCapture(str(input_path))
            debug_cap.set(cv2.CAP_PROP_POS_FRAMES, args.skip)
            debug_ok, debug_frame = debug_cap.read()
            debug_cap.release()
            
            if debug_ok:
                debug_binary = threshold_lane_pixels(debug_frame, preprocess_params)
                debug_mask = (debug_binary * 255).astype(np.uint8)
                debug_warped = warp_image(debug_mask, perspective_matrix)
                debug_warped_binary = (debug_warped > 0).astype(np.uint8)
                
                debug_detection = sliding_window_fit(
                    debug_warped_binary,
                    params=search_params,
                    prev_left_fit=None,
                    prev_right_fit=None,
                )
                
                save_debug_images(debug_frame, 0, debug_binary, debug_warped_binary, 
                                debug_detection, debug_dir, preprocess_params)
                print(f"  Debug images saved to: {debug_dir}/")
        
        # Generate analysis plots after video processing
        print("\nGenerating analysis plots...")
        from .analysis import _read_csv, summarize_run, plot_run
        data = _read_csv(csv_path)
        summary = summarize_run(data)
        plot_path = plot_run(data, csv_path.parent, title=csv_path.stem)
        
        print(f"\nAnalysis complete:")
        print(f"  Frames processed: {summary.total_frames}")
        print(f"  Left detection rate: {summary.left_detection_rate:.2%}")
        print(f"  Right detection rate: {summary.right_detection_rate:.2%}")
        print(f"  Mean conf (L/R): {summary.mean_conf_left:.2f} / {summary.mean_conf_right:.2f}")
        if summary.lateral_offset_mean is not None:
            print(f"  Lateral offset: {summary.lateral_offset_mean:.2f} ± {summary.lateral_offset_std:.2f} m")
        print(f"  Plot saved to: {plot_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output_data) if args.output_data else Path("outputs_data") / f"{input_path.stem}_annotated.mp4"
    csv_path = Path(args.csv) if args.csv else Path("outputs_data") / f"{input_path.stem}_per_frame.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    process_video(input_path, output_path, csv_path, args)


if __name__ == "__main__":
    main()
