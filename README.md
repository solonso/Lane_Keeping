# Name: Solomon Chibuzo Nwafor 
# Neptun ID: UITK8W
# LKA - Line Keep Assist Implementation

Classical Computer Vision approach to Lane Keep Assist (LKA) for ADAS using video analysis.

## Overview

This implementation detects left and right lane boundaries in driving videos using classical image processing techniques. For each frame, it outputs:
- Binary indicators for left/right lane detection
- Lane curves (polylines) overlaid on the road
- Per-frame confidence scores and lateral offset

## Project Structure

```
LKA_submission/
├── main/                          # Input videos
│   ├── challenge.mp4             # Challenge dataset with shadows/curves
│   └── lane.mp4                  # Standard highway driving
├── outputs_data/                 # Generated outputs
│   ├── challenge_annotated.mp4   # Annotated challenge video
│   ├── challenge_per_frame.csv   # Per-frame metrics
│   ├── challenge_per_frame_run.png  # Metrics visualization
│   ├── challenge_debug_frames/   # Debug images showing pipeline stages
│   ├── lane_annotated.mp4        # Annotated lane video
│   ├── lane_per_frame.csv        # Per-frame metrics
│   ├── lane_per_frame_run.png    # Metrics visualization
│   └── lane_debug_frames/        # Debug images showing pipeline stages
├── src/                          # Source code
│   ├── main.py                   # Main entry point
│   ├── preprocess.py             # Color & gradient thresholding
│   ├── warp.py                   # Perspective transform (IPM)
│   ├── lane_fit.py               # Sliding window & polynomial fitting
│   ├── temporal.py               # Temporal smoothing
│   ├── overlay.py                # Visualization & HUD
│   ├── csv_writer.py             # CSV logging
│   └── analysis.py               # Metrics analysis
├── requirements.txt              # Python dependencies
└── README.md                     # This file (you are here)
```

## Installation

1. Ensure Python 3.8+ is installed:
   ```bash
   python3 --version
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Process Videos

Run the lane detection pipeline on a video:

```bash
python -m src.main --input main/lane.mp4
```

This will generate:
- `outputs_data/lane_annotated.mp4` - Video with lane overlays
- `outputs_data/lane_per_frame.csv` - Per-frame detection metrics
- `outputs_data/lane_per_frame_run.png` - Metrics visualization
- `outputs_data/lane_debug_frames/` - Debug images showing pipeline stages

For the challenge video:

```bash
python -m src.main --input main/challenge.mp4
```

This generates:
- `outputs_data/challenge_annotated.mp4`
- `outputs_data/challenge_per_frame.csv`
- `outputs_data/challenge_per_frame_run.png` - Metrics visualization
- `outputs_data/challenge_debug_frames/` - Debug images showing pipeline stages

Note: Analysis plots and debug frames are automatically generated after processing completes. No separate command needed.

## Pipeline Overview

### 1. Preprocessing (`preprocess.py`)
- **ROI Selection**: Trapezoidal region of interest (lower 68% of frame)
- **Color Thresholding**: HSV/HLS color space analysis
  - Yellow lanes: H=15-35°, S≥80, V≥120
  - White lanes: S≤40, V≥180
- **Gradient Thresholding**: Sobel-X operator on L-channel (kernel=3)
- **Morphological Operations**: Closing and opening to reduce noise
- **Connected Components**: Remove artifacts <50 pixels

### 2. Perspective Transform (`warp.py`)
- **Inverse Perspective Mapping (IPM)**: Transform to bird's-eye view
- Source points: (42%, 58%) width at 62% height, (10%, 90%) at bottom
- Destination: Rectangular view for easier lane fitting

### 3. Lane Detection (`lane_fit.py`)
- **Histogram Analysis**: Find lane base positions
- **Sliding Windows**: 9 windows, 80px margin, 40px minimum pixels
- **Polynomial Fitting**: 2nd-order polynomial x = f(y)
- **Geometric Validation**: 
  - Lane width: 150-900 pixels
  - Width variation: <120% (curves)
  - Parallelism check
- **Confidence Calculation**:
  ```
  confidence = 0.6 * pixel_count_score + 0.25 * residual_score + 0.15 * prior_score
  ```
- **Detection Threshold**: 0.60

### 4. Temporal Smoothing (`temporal.py`)
- **Exponential Blending**: α=0.2
- **Persistence**: Reuse previous fit for up to 6 frames if conf≥0.40
- **Decay Factor**: 0.90 per frame when reusing

### 5. Visualization (`overlay.py`)
- **Left lane**: Green polyline
- **Right lane**: Blue polyline
- **Low confidence**: Dashed gray line
- **HUD Display**:
  - Top-left: Lane confidence and detection status
  - Top-right: Detection stats (rate, flickers per 10s)
  - Lateral offset in meters

### 6. Metrics (`analysis.py`)
- **Detection Rate**: % of frames with successful detection
- **Mean Confidence**: Average confidence scores
- **Lateral Offset**: Mean and std deviation in meters
- **Curvature**: Estimated road curvature
- **Visualizations**: Time-series plots of all metrics

## Output Format

### CSV Structure
```csv
frame_id,left_detected,right_detected,left_conf,right_conf,lat_offset_m,left_curv_m,right_curv_m
0,1,1,0.92,0.90,-0.12,,284.64
1,1,0,0.88,0.42,-0.10,,329.02
```

### Video Overlays
- **Solid colored line**: Confidently detected lane
- **Dashed gray line**: Low confidence detection
- **Green fill**: Ego-lane corridor
- **HUD panels**: Real-time detection statistics

## Results

### Lane Dataset (lane.mp4)
- Detection rate: 100% (left), 100% (right)
- Mean confidence: 0.94 (both lanes)
- Lateral offset: -0.15 ± 0.08 m
- Flickers per 10s: <0.5

### Challenge Dataset (challenge.mp4)
- Detection rate: 96.47% (left), 99.29% (right)
- Mean confidence: 0.91 (left), 0.92 (right)
- Lateral offset: -0.09 ± 0.11 m
- Flickers per 10s: ~1.0

### Key Scenarios
1. **Stable Highway**: Robust detection with >90% confidence
2. **Shadow Environments**: Partial degradation (frames 129-137)
3. **Curved Roads**: Stable with curvature compensation

### Debug Frames
Each video processing run automatically generates debug images in `{video_name}_debug_frames/` showing the pipeline stages:
1. **01_roi.jpg**: Original frame with ROI mask highlighted
2. **02a_L_channel.jpg**: Luminance channel (L from HLS)
3. **02b_S_channel.jpg**: Saturation channel (S from HLS)
4. **02c_V_channel.jpg**: Value channel (V from HSV)
5. **02d_gray.jpg**: Grayscale conversion
6. **03_binary_mask.jpg**: Combined binary mask after thresholding
7. **04_warped_binary.jpg**: Bird's-eye view after IPM transformation
8. **05_histogram.jpg**: Lane base histogram with detected peaks
9. **06_detected_lanes.jpg**: Fitted polynomial lanes overlaid on warped view

These images help visualize each processing stage for debugging and report documentation.

## Algorithm Details

### Confidence Scoring
Three-factor model:
1. **Pixel Count**: Normalized by target (1000 pixels)
2. **Fit Residual**: Average distance to polynomial
3. **Temporal Prior**: Deviation from previous fit

### Temporal Filtering
- New detection: Blend with previous (α=0.2)
- Missing detection: Reuse prior if conf≥0.40
- Persistence limit: 6 frames before giving up

### Geometric Constraints
- Minimum lane width: 150 px (narrow lanes rejected)
- Maximum lane width: 900 px (merges/rejects false pairs)
- Width variation: <120% (enforces parallelism)
- Low coverage: Fall back to linear fit if <40%

## Failure Modes

1. **Shadows**: Reduced contrast affects color thresholds
2. **Faded Paint**: Gradient-only detection becomes noisy
3. **Sharp Curves**: Width validation may reject valid lanes
4. **Camera Calibration**: Fixed IPM assumes standard mounting

## Technical Stack

- **OpenCV 4.10**: Image processing, video I/O
- **NumPy 1.24**: Numerical operations, polynomial fitting
- **Matplotlib 3.1**: Metrics visualization

## Limitations

- No camera calibration (assumes fixed mounting)
- Curvature estimation unreliable (many NaN values)
- Limited to daytime, good weather conditions
- Fixed IPM parameters not adaptable to all cameras
- No ground truth validation

## Future Improvements

- Adaptive thresholds based on illumination
- Camera calibration for accurate IPM
- Improved curvature estimation
- Machine learning for robustness
- Night/rain domain adaptation
- Real-time performance profiling