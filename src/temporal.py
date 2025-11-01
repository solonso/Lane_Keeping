import numpy as np

from .lane_fit import LaneDetection


class TemporalParams:
    def __init__(self, alpha=0.2, min_reuse_conf=0.40, detect_threshold=0.48, persistence=6, reuse_decay=0.90):
        self.alpha = alpha
        self.min_reuse_conf = min_reuse_conf
        self.detect_threshold = detect_threshold
        self.persistence = persistence
        self.reuse_decay = reuse_decay


class LaneTemporalFilter:
    def __init__(self, params: TemporalParams):
        self.params = params
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.prev_left_conf = 0.0
        self.prev_right_conf = 0.0
        self.left_miss_count = 0
        self.right_miss_count = 0

    def _blend(
        self,
        current_fit,
        prev_fit,
        alpha,
    ):
        if current_fit is None and prev_fit is None:
            return None
        if current_fit is None:
            return prev_fit
        if prev_fit is None:
            return current_fit
        return alpha * current_fit + (1.0 - alpha) * prev_fit

    def apply(self, detection: LaneDetection):
        alpha = self.params.alpha

        if detection.left.coeffs is not None:
            blended_left = self._blend(detection.left.coeffs, self.prev_left_fit, alpha)
            detection.left.coeffs = blended_left
            if detection.left.coeffs is not None:
                detection.left.confidence = max(detection.left.confidence, self.prev_left_conf * (1.0 - alpha))
            detection.left.detected = detection.left.confidence >= self.params.detect_threshold
            self.left_miss_count = 0
        elif self.prev_left_fit is not None and self.prev_left_conf >= self.params.min_reuse_conf and self.left_miss_count < self.params.persistence:
            detection.left.coeffs = self.prev_left_fit.copy()
            detection.left.confidence = max(detection.left.confidence, self.prev_left_conf * self.params.reuse_decay)
            detection.left.detected = detection.left.confidence >= self.params.detect_threshold
            self.left_miss_count += 1
        else:
            detection.left.coeffs = None
            detection.left.confidence = 0.0
            detection.left.detected = False
            self.left_miss_count = self.params.persistence + 1

        if detection.right.coeffs is not None:
            blended_right = self._blend(detection.right.coeffs, self.prev_right_fit, alpha)
            detection.right.coeffs = blended_right
            if detection.right.coeffs is not None:
                detection.right.confidence = max(detection.right.confidence, self.prev_right_conf * (1.0 - alpha))
            detection.right.detected = detection.right.confidence >= self.params.detect_threshold
            self.right_miss_count = 0
        elif self.prev_right_fit is not None and self.prev_right_conf >= self.params.min_reuse_conf and self.right_miss_count < self.params.persistence:
            detection.right.coeffs = self.prev_right_fit.copy()
            detection.right.confidence = max(detection.right.confidence, self.prev_right_conf * self.params.reuse_decay)
            detection.right.detected = detection.right.confidence >= self.params.detect_threshold
            self.right_miss_count += 1
        else:
            detection.right.coeffs = None
            detection.right.confidence = 0.0
            detection.right.detected = False
            self.right_miss_count = self.params.persistence + 1

        self.prev_left_fit = detection.left.coeffs.copy() if detection.left.coeffs is not None else None
        self.prev_right_fit = detection.right.coeffs.copy() if detection.right.coeffs is not None else None
        self.prev_left_conf = detection.left.confidence
        self.prev_right_conf = detection.right.confidence

        return detection
