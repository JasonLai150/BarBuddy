"""
Stage 1 — Person Detection (YOLOv8).

Detects person bounding boxes in each frame and provides simple
IoU-based tracking to maintain consistent person IDs across frames.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Person detector
# ---------------------------------------------------------------------------
class PersonDetector:
    """YOLOv8-based single/multi-person detector.

    Usage:
        detector = PersonDetector(device="cuda", confidence=0.5)
        for frame in frames:
            bboxes = detector.detect(frame)
            # bboxes: list of dicts {x1, y1, x2, y2, conf, person_id}
    """

    # Available sizes: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    DEFAULT_MODEL = "yolov8m.pt"

    def __init__(
        self,
        device: str = "cuda",
        confidence: float = 0.5,
        model_name: str | None = None,
        iou_threshold: float = 0.45,
    ):
        model_name = model_name or self.DEFAULT_MODEL
        self.model = YOLO(model_name)
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Simple tracker state
        self._next_id = 0
        self._prev_boxes: list[dict] = []

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect person bounding boxes in a single BGR image.

        Args:
            image: np.ndarray of shape (H, W, 3), BGR colour order.

        Returns:
            List of dicts with keys:
                x1, y1, x2, y2   – pixel coordinates
                conf              – detection confidence [0, 1]
                person_id         – tracked ID (stable across frames)
        """
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[0],  # COCO class 0 = "person"
        )

        detections: list[dict] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "conf": float(box.conf[0]),
                    "person_id": -1,  # assigned below by tracker
                })

        # Sort by confidence (highest first) then assign tracked IDs
        detections.sort(key=lambda d: d["conf"], reverse=True)
        detections = self._assign_ids(detections)
        self._prev_boxes = detections
        return detections

    def select_primary(self, bboxes: list[dict]) -> dict | None:
        """
        Select the *primary* person (largest + most central bounding box).

        For barbell lifting analysis we usually care about exactly one lifter.
        Heuristic: score = area × (1 − distance_from_center).
        """
        if not bboxes:
            return None
        if len(bboxes) == 1:
            return bboxes[0]

        best, best_score = None, -1.0
        for b in bboxes:
            area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            # No image dims needed — area in pixels is a fine proxy
            score = area * b["conf"]
            if score > best_score:
                best_score = score
                best = b
        return best

    # ------------------------------------------------------------------
    # Simple IoU tracker (no Kalman, just greedy matching)
    # ------------------------------------------------------------------
    def _assign_ids(self, current: list[dict]) -> list[dict]:
        if not self._prev_boxes:
            # First frame — assign fresh IDs
            for d in current:
                d["person_id"] = self._next_id
                self._next_id += 1
            return current

        prev = self._prev_boxes
        cost = _iou_matrix(prev, current)  # (N_prev, N_curr)

        matched_prev: set[int] = set()
        matched_curr: set[int] = set()

        # Greedy matching: iterate IoU pairs in descending order
        if cost.size > 0:
            flat = cost.flatten()
            order = np.argsort(-flat)
            for idx in order:
                pi = int(idx // cost.shape[1])
                ci = int(idx % cost.shape[1])
                if pi in matched_prev or ci in matched_curr:
                    continue
                if cost[pi, ci] < 0.3:  # IoU threshold for matching
                    break
                current[ci]["person_id"] = prev[pi]["person_id"]
                matched_prev.add(pi)
                matched_curr.add(ci)

        # Assign new IDs to unmatched detections
        for ci, d in enumerate(current):
            if ci not in matched_curr:
                d["person_id"] = self._next_id
                self._next_id += 1

        return current


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------
def _iou_matrix(boxes_a: list[dict], boxes_b: list[dict]) -> np.ndarray:
    """Compute pairwise IoU between two lists of bbox dicts."""
    a = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_a])
    b = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_b])
    return _iou_np(a, b)


def _iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between arrays of shape (N,4) and (M,4)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)
