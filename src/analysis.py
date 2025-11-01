import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class RunSummary:
    def __init__(
        self,
        total_frames: int,
        left_detection_rate: float,
        right_detection_rate: float,
        mean_conf_left: float,
        mean_conf_right: float,
        lateral_offset_mean: Optional[float],
        lateral_offset_std: Optional[float],
    ):
        self.total_frames = total_frames
        self.left_detection_rate = left_detection_rate
        self.right_detection_rate = right_detection_rate
        self.mean_conf_left = mean_conf_left
        self.mean_conf_right = mean_conf_right
        self.lateral_offset_mean = lateral_offset_mean
        self.lateral_offset_std = lateral_offset_std


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float]] = []
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No data found in {path}")
    columns: Dict[str, List[str]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            columns[key].append(value)
    result: Dict[str, np.ndarray] = {}
    for key, values in columns.items():
        parsed: List[float] = []
        fallback: List[str] = []
        numeric = True
        for value in values:
            if value == "" or value is None:
                parsed.append(float("nan"))
                continue
            try:
                parsed.append(float(value))
            except ValueError:
                numeric = False
                break
        if numeric:
            result[key] = np.array(parsed, dtype=float)
        else:
            fallback = [str(v) for v in values]
            result[key] = np.array(fallback)
    return result


def summarize_run(data: Dict[str, np.ndarray]) -> RunSummary:
    total = int(len(data["frame_id"]))
    left_det = float(np.mean(data["left_detected"])) if total else 0.0
    right_det = float(np.mean(data["right_detected"])) if total else 0.0
    mean_conf_left = float(np.mean(data["left_conf"])) if total else 0.0
    mean_conf_right = float(np.mean(data["right_conf"])) if total else 0.0

    lat = data.get("lat_offset_m")
    lat_clean = None
    lat_std = None
    if lat is not None:
        valid = ~np.isnan(lat)
        if valid.any():
            lat_clean = float(np.mean(lat[valid]))
            lat_std = float(np.std(lat[valid]))
    return RunSummary(
        total_frames=total,
        left_detection_rate=left_det,
        right_detection_rate=right_det,
        mean_conf_left=mean_conf_left,
        mean_conf_right=mean_conf_right,
        lateral_offset_mean=lat_clean,
        lateral_offset_std=lat_std,
    )


def plot_run(data: Dict[str, np.ndarray], output_dir: Path, title: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = data["frame_id"]

    plt.style.use("seaborn-v0_8") if "seaborn-v0_8" in plt.style.available else None
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(frame_ids, data["left_conf"], label="Left", color="#2ecc71")
    axes[0].plot(frame_ids, data["right_conf"], label="Right", color="#3498db")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    axes[0].set_title(f"Lane Confidence — {title}")

    if "lat_offset_m" in data:
        axes[1].plot(frame_ids, data["lat_offset_m"], color="#f1c40f")
        axes[1].set_ylabel("Offset (m)")
        axes[1].axhline(0.0, color="#95a5a6", linewidth=1.0, linestyle="--")
    else:
        axes[1].set_visible(False)

    if "left_curv_m" in data and "right_curv_m" in data:
        axes[2].plot(frame_ids, data["left_curv_m"], color="#9b59b6", label="Left curvature")
        axes[2].plot(frame_ids, data["right_curv_m"], color="#e74c3c", label="Right curvature")
        axes[2].set_ylabel("Curvature (m)")
        axes[2].legend(loc="upper right")
    else:
        axes[2].set_visible(False)

    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    output_path = output_dir / f"{title.replace(' ', '_').lower()}_run.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Summarise and visualise LKA run metrics.")
    parser.add_argument("--csv", required=True, help="Per-frame CSV produced by the pipeline.")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory to save plots.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    data = _read_csv(csv_path)
    summary = summarize_run(data)
    plot_path = plot_run(data, output_dir, title=csv_path.stem)

    print(f"Frames processed: {summary.total_frames}")
    print(f"Left detection rate: {summary.left_detection_rate:.2%}")
    print(f"Right detection rate: {summary.right_detection_rate:.2%}")
    print(f"Mean conf (L/R): {summary.mean_conf_left:.2f} / {summary.mean_conf_right:.2f}")
    if summary.lateral_offset_mean is not None:
        print(
            f"Lateral offset mean ± std: {summary.lateral_offset_mean:.2f} ± "
            f"{summary.lateral_offset_std:.2f} m"
        )
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    run_cli()
