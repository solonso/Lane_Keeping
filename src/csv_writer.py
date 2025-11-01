import csv
from pathlib import Path
from typing import Optional


class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.file = None
        self.writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "frame_id",
                "left_detected",
                "right_detected",
                "left_conf",
                "right_conf",
                "lat_offset_m",
                "left_curv_m",
                "right_curv_m",
            ]
        )
        return self

    def write(
        self,
        frame_id: int,
        left_detected: bool,
        right_detected: bool,
        left_conf: float,
        right_conf: float,
        lateral_offset_m: Optional[float],
        left_curvature_m: Optional[float],
        right_curvature_m: Optional[float],
    ) -> None:
        if self.writer is None:
            raise RuntimeError("CSVLogger must be used as a context manager.")
        self.writer.writerow(
            [
                frame_id,
                int(left_detected),
                int(right_detected),
                round(left_conf, 4),
                round(right_conf, 4),
                round(lateral_offset_m, 4) if lateral_offset_m is not None else "",
                round(left_curvature_m, 2) if left_curvature_m is not None else "",
                round(right_curvature_m, 2) if right_curvature_m is not None else "",
            ]
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.flush()
            self.file.close()
