"""Ball detection over a video using TrackNet + Hough circle localization."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from padelvision.config import DetectionConfig
from padelvision.detection.tracknet import build_tracknet

# frame index -> (x, y) pixel coordinate, or None when no ball is found.
Detections = Dict[int, Optional[Tuple[int, int]]]


def locate_ball(heatmap: np.ndarray, config: DetectionConfig) -> Optional[Tuple[int, int]]:
    """Find a single ball centre in a model heatmap, or None.

    Pure OpenCV/NumPy (no Keras) so it can be unit-tested in isolation.
    """
    _, binary = cv2.threshold(heatmap, config.binary_threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT,
        dp=config.hough_dp, minDist=config.hough_min_dist,
        param1=config.hough_param1, param2=config.hough_param2,
        minRadius=config.hough_min_radius, maxRadius=config.hough_max_radius,
    )
    if circles is not None and len(circles) == 1:
        x = int(circles[0][0][0])
        y = int(circles[0][0][1])
        return x, y
    return None


class BallTracker:
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self._model = None  # built lazily on first use

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        model = build_tracknet(
            self.config.n_classes,
            input_height=self.config.height,
            input_width=self.config.width,
        )
        model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
        model.load_weights(self.config.weights_path)
        return model

    def _predict_heatmap(self, frames: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
        """Run the model on 3 stacked frames and return an upscaled uint8 heatmap."""
        pr = self.model.predict(np.array([frames]))[0]
        pr = pr.reshape((self.config.height, self.config.width, self.config.n_classes)).argmax(axis=2)
        pr = pr.astype(np.uint8)
        return cv2.resize(pr, out_size)

    def detect(
        self,
        input_video_path: str,
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Detections:
        """Detect the ball in every frame of a video.

        If ``read_from_stub`` is set and ``stub_path`` exists, cached detections
        are returned instead of re-running the model. When ``stub_path`` is set
        the freshly computed detections are written there.
        """
        if read_from_stub and stub_path and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        video = cv2.VideoCapture(input_video_path)
        out_size = (
            int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        detections: Detections = {}

        def read_resized() -> Optional[np.ndarray]:
            ok, frame = video.read()
            if not ok:
                return None
            frame = cv2.resize(frame, (self.config.width, self.config.height))
            return frame.astype(np.float32)

        prev2 = read_resized()
        prev1 = read_resized()
        frame_index = 2
        while prev1 is not None and prev2 is not None:
            current = read_resized()
            if current is None:
                break
            stacked = np.concatenate((current, prev1, prev2), axis=2)
            stacked = np.rollaxis(stacked, 2, 0)
            heatmap = self._predict_heatmap(stacked, out_size)
            detections[frame_index] = locate_ball(heatmap, self.config)
            prev2, prev1 = prev1, current
            frame_index += 1

        video.release()

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(detections, f)
        return detections
