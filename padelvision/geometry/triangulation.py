"""Triangulate paired 2D ball detections into 3D points.

Unlike the original prototype this works from real projection matrices, which
can be built directly from the ``stereo_params.yml`` produced by
:class:`padelvision.calibration.stereo.StereoCalibrator`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def matched_points(
    det1: Dict[int, Optional[Tuple[int, int]]],
    det2: Dict[int, Optional[Tuple[int, int]]],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Align two per-frame detection dicts into paired point arrays.

    Keeps only frames where *both* cameras found the ball. Returns
    ``(pts1, pts2, frames)`` where ``pts1``/``pts2`` are (N, 2) arrays ready for
    :meth:`Triangulator.triangulate` and ``frames`` lists the frame indices used.
    """
    frames = [
        f for f in sorted(det1.keys() & det2.keys())
        if det1[f] is not None and det2[f] is not None
    ]
    pts1 = np.array([det1[f] for f in frames], dtype=float).reshape(-1, 2)
    pts2 = np.array([det2[f] for f in frames], dtype=float).reshape(-1, 2)
    return pts1, pts2, frames


def projection_matrix(K: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Build a 3x4 projection matrix ``P = K [R | T]``."""
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    T = np.asarray(T, dtype=float).reshape(3, 1)
    return K @ np.hstack((R, T))


class Triangulator:
    """Triangulate corresponding 2D points seen by two calibrated cameras."""

    def __init__(self, P1: np.ndarray, P2: np.ndarray):
        self.P1 = np.asarray(P1, dtype=float)
        self.P2 = np.asarray(P2, dtype=float)
        if self.P1.shape != (3, 4) or self.P2.shape != (3, 4):
            raise ValueError("Projection matrices must be 3x4")

    @classmethod
    def from_projection_matrices(cls, P1: np.ndarray, P2: np.ndarray) -> "Triangulator":
        return cls(P1, P2)

    @classmethod
    def from_stereo_params(cls, path: str | Path) -> "Triangulator":
        """Build a triangulator from an OpenCV ``stereo_params.yml`` file.

        Camera 1 is treated as the reference frame (P1 = K1 [I | 0]); camera 2
        uses the relative rotation/translation (P2 = K2 [R | T]).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stereo params not found: {path}")
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            K1 = fs.getNode("K1").mat()
            K2 = fs.getNode("K2").mat()
            R = fs.getNode("R").mat()
            T = fs.getNode("T").mat()
        finally:
            fs.release()
        if any(m is None for m in (K1, K2, R, T)):
            raise ValueError(f"{path} is missing one of K1/K2/R/T")
        P1 = projection_matrix(K1, np.eye(3), np.zeros((3, 1)))
        P2 = projection_matrix(K2, R, T)
        return cls(P1, P2)

    def triangulate(
        self,
        points_cam1: Sequence[Sequence[float]] | np.ndarray,
        points_cam2: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Triangulate matched points.

        Args:
            points_cam1: (N, 2) pixel coordinates in camera 1.
            points_cam2: (N, 2) pixel coordinates in camera 2.

        Returns:
            (N, 3) array of 3D points.
        """
        pts1 = np.asarray(points_cam1, dtype=float).reshape(-1, 2)
        pts2 = np.asarray(points_cam2, dtype=float).reshape(-1, 2)
        if pts1.shape != pts2.shape:
            raise ValueError("Both cameras must provide the same number of points")
        if len(pts1) == 0:
            return np.empty((0, 3))
        # cv2.triangulatePoints expects 2xN float arrays.
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        return points_3d
