"""
Stage 2 — 2D Pose Estimation.

Supports two backends:
  • "rtmpose"  — RTMPose-m via ONNX Runtime (top-down, more accurate under occlusion)
  • "yolov8"   — YOLOv8-Pose via ultralytics  (bottom-up, simpler, good fallback)

Both produce COCO-17 keypoints normalised to [0, 1] relative to the full image.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from src.utils import COCO_KEYPOINTS, download_model


# ===================================================================
# RTMPose backend (ONNX Runtime)
# ===================================================================
# RTMPose-m body model — COCO 256×192, SimCC head
_RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
_RTMPOSE_ONNX_FILENAME = "rtmpose-m.onnx"

# Model input dimensions (width, height)
_MODEL_INPUT_SIZE = (192, 256)

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _RTMPoseBackend:
    """RTMPose inference via ONNX Runtime with manual pre/post-processing."""

    def __init__(self, device: str = "cuda", model_path: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for RTMPose backend. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime for CPU)"
            )

        if model_path is None:
            model_path = self._ensure_model()

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _ensure_model() -> str:
        """Download and extract RTMPose ONNX model if not cached."""
        from src.utils import MODEL_CACHE_DIR

        onnx_path = os.path.join(MODEL_CACHE_DIR, _RTMPOSE_ONNX_FILENAME)
        if os.path.isfile(onnx_path):
            return onnx_path

        # Download the zip, extract the .onnx file
        zip_path = download_model(_RTMPOSE_URL, filename="rtmpose-m.zip")

        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the .onnx file inside the archive
            onnx_members = [m for m in zf.namelist() if m.endswith(".onnx")]
            if not onnx_members:
                raise RuntimeError(f"No .onnx file found in {zip_path}")
            zf.extract(onnx_members[0], MODEL_CACHE_DIR)
            extracted = os.path.join(MODEL_CACHE_DIR, onnx_members[0])
            if extracted != onnx_path:
                os.rename(extracted, onnx_path)

        return onnx_path

    # ----- Pre-processing -----
    @staticmethod
    def _box_to_center_scale(bbox: dict, padding: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
        """Convert bbox dict to (center, scale) for affine warp."""
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

        w = x2 - x1
        h = y2 - y1
        aspect = _MODEL_INPUT_SIZE[0] / _MODEL_INPUT_SIZE[1]  # 192/256 = 0.75

        if w > aspect * h:
            h = w / aspect
        elif w < aspect * h:
            w = h * aspect

        scale = np.array([w, h], dtype=np.float32) * padding
        return center, scale

    @staticmethod
    def _get_affine_transform(center: np.ndarray, scale: np.ndarray, output_size: tuple[int, int], inv: bool = False):
        """Compute the affine transform from (center, scale) → model input."""
        src_w = scale[0]
        dst_w, dst_h = output_size

        src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)

        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        # Third point perpendicular to first two
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            return cv2.getAffineTransform(dst, src)
        return cv2.getAffineTransform(src, dst)

    def _preprocess(self, image: np.ndarray, bbox: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop and normalise image for RTMPose.

        Returns:
            blob:   (1, 3, 256, 192) float32 tensor
            center: (2,) array
            scale:  (2,) array
        """
        center, scale = self._box_to_center_scale(bbox)
        trans = self._get_affine_transform(center, scale, _MODEL_INPUT_SIZE)

        crop = cv2.warpAffine(
            image, trans, _MODEL_INPUT_SIZE,
            flags=cv2.INTER_LINEAR,
        )

        # Normalise: RGB, /255, ImageNet mean/std
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - _MEAN) / _STD
        blob = crop.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        return blob.astype(np.float32), center, scale

    # ----- Post-processing (SimCC decode) -----
    def _postprocess(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> list[dict]:
        """
        Decode SimCC outputs to normalised keypoints.

        Args:
            simcc_x: (1, 17, Wx) logits for x-coordinates
            simcc_y: (1, 17, Wy) logits for y-coordinates

        Returns:
            List of 17 keypoint dicts {idx, name, x, y, conf}
        """
        # Argmax → sub-pixel position in crop space
        x_locs = np.argmax(simcc_x[0], axis=-1).astype(np.float32) / 2.0  # (17,)
        y_locs = np.argmax(simcc_y[0], axis=-1).astype(np.float32) / 2.0  # (17,)

        # Confidence from SimCC logits.
        # Softmax peak is meaningless over 384/512 bins (~0.007), so we use
        # the logit spread: (max - mean) passed through a sigmoid gives a
        # [0, 1] confidence that reflects how peaked the distribution is.
        def _logit_confidence(logits: np.ndarray) -> np.ndarray:
            """Per-joint confidence from logit spread (max − mean) → sigmoid."""
            max_vals = logits.max(axis=-1)      # (17,)
            mean_vals = logits.mean(axis=-1)    # (17,)
            spread = max_vals - mean_vals       # how peaked
            return 1.0 / (1.0 + np.exp(-spread))  # sigmoid → [0, 1]

        x_conf = _logit_confidence(simcc_x[0])  # (17,)
        y_conf = _logit_confidence(simcc_y[0])  # (17,)
        confs = np.sqrt(x_conf * y_conf)  # geometric mean

        # Inverse affine: crop coords → image coords
        inv_trans = self._get_affine_transform(center, scale, _MODEL_INPUT_SIZE, inv=True)

        keypoints: list[dict] = []
        for i in range(len(x_locs)):
            pt_crop = np.array([x_locs[i], y_locs[i], 1.0], dtype=np.float32)
            pt_img = inv_trans @ pt_crop  # (2,)

            keypoints.append({
                "idx": i,
                "name": COCO_KEYPOINTS[i],
                "x": float(np.clip(pt_img[0] / img_w, 0.0, 1.0)),
                "y": float(np.clip(pt_img[1] / img_h, 0.0, 1.0)),
                "conf": float(confs[i]),
            })

        return keypoints

    def estimate(self, image: np.ndarray, bboxes: list[dict]) -> list[dict] | None:
        """
        Run 2D pose estimation for the primary person.

        Args:
            image:  BGR image (H, W, 3)
            bboxes: list of person bboxes from Stage 1

        Returns:
            List of 17 COCO keypoint dicts, or None if no person detected.
        """
        if not bboxes:
            return None

        h, w = image.shape[:2]
        bbox = bboxes[0]  # primary person (already sorted by detector)

        blob, center, scale = self._preprocess(image, bbox)
        outputs = self.session.run(None, {self.input_name: blob})

        # RTMPose SimCC outputs: simcc_x, simcc_y
        simcc_x, simcc_y = outputs[0], outputs[1]
        return self._postprocess(simcc_x, simcc_y, center, scale, w, h)


# ===================================================================
# YOLOv8-Pose backend (simpler fallback)
# ===================================================================
class _YOLOv8PoseBackend:
    """YOLOv8-Pose for combined detection + 2D pose in one pass."""

    def __init__(self, device: str = "cuda", model_name: str = "yolov8m-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.device = device

    def estimate(self, image: np.ndarray, bboxes: list[dict] | None = None) -> list[dict] | None:
        """
        Run pose estimation.  Ignores bboxes — YOLOv8-Pose does its own detection.

        Returns:
            List of 17 COCO keypoint dicts, or None.
        """
        h, w = image.shape[:2]
        results = self.model(image, device=self.device, verbose=False)

        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            # Take the most confident person
            kpts = result.keypoints[0]  # first person
            data = kpts.data[0].cpu().numpy()  # (17, 3) → x, y, conf

            keypoints: list[dict] = []
            for i in range(data.shape[0]):
                keypoints.append({
                    "idx": i,
                    "name": COCO_KEYPOINTS[i],
                    "x": float(np.clip(data[i, 0] / w, 0.0, 1.0)),
                    "y": float(np.clip(data[i, 1] / h, 0.0, 1.0)),
                    "conf": float(data[i, 2]),
                })
            return keypoints

        return None


# ===================================================================
# Public API
# ===================================================================
class PoseEstimator2D:
    """
    Unified 2D pose estimator with configurable backend.

    Usage:
        pose = PoseEstimator2D(backend="rtmpose", device="cuda")
        keypoints = pose.estimate(image, bboxes)
    """

    BACKENDS = {"rtmpose", "yolov8"}

    def __init__(
        self,
        backend: str = "rtmpose",
        device: str = "cuda",
        model_path: str | None = None,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {self.BACKENDS}")

        self.backend_name = backend
        if backend == "rtmpose":
            self._backend = _RTMPoseBackend(device=device, model_path=model_path)
        else:
            self._backend = _YOLOv8PoseBackend(device=device)

    def estimate(self, image: np.ndarray, bboxes: list[dict] | None = None) -> list[dict] | None:
        """
        Estimate 2D keypoints for the primary person.

        Args:
            image:  BGR frame (H, W, 3)
            bboxes: person bounding boxes from Stage 1 (used by rtmpose backend)

        Returns:
            List of 17 dicts {idx, name, x, y, conf} or None
        """
        return self._backend.estimate(image, bboxes)


# ===================================================================
# Helpers
# ===================================================================
def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the third point to define a unique affine transform."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
