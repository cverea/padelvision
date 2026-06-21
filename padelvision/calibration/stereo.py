"""Stereo calibration from chessboard image pairs.

Refactor of the original ``stereo_calibration.py``: configuration-driven,
GUI display is opt-in (off by default so it can run headless / in tests), and
calibration results are returned as a structured object as well as optionally
written to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from padelvision.config import CalibrationConfig


@dataclass
class StereoCalibrationResult:
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    rms: float
    image_size: tuple[int, int]

    def save(self, path: str | Path) -> None:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        try:
            for name in ("K1", "D1", "K2", "D2", "R", "T", "Q"):
                fs.write(name, getattr(self, name))
        finally:
            fs.release()


class StereoCalibrator:
    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.config.criteria_max_iter,
            self.config.criteria_epsilon,
        )
        # Object points for one board (Z=0 plane), scaled to real-world size.
        cols, rows = self.config.chessboard_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        self._objp = objp * self.config.square_size

    def _image_paths(self, folder: str) -> List[Path]:
        return sorted(Path(folder).glob(self.config.image_glob))

    def find_corners(self, show: bool = False):
        """Detect chessboard corners in every left/right image pair.

        Returns (objpoints, imgpoints_left, imgpoints_right, image_size).
        """
        left = self._image_paths(self.config.left_dir)
        right = self._image_paths(self.config.right_dir)
        if len(left) != len(right):
            raise ValueError(
                f"Image count mismatch: {len(left)} in {self.config.left_dir} "
                f"vs {len(right)} in {self.config.right_dir}"
            )
        if not left:
            raise ValueError(f"No images found in {self.config.left_dir}")

        objpoints: list = []
        imgpoints_left: list = []
        imgpoints_right: list = []
        image_size: tuple[int, int] | None = None
        win = (self.config.corner_window, self.config.corner_window)

        for lp, rp in zip(left, right):
            img_l = cv2.imread(str(lp))
            img_r = cv2.imread(str(rp))
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            image_size = gray_l.shape[::-1]
            ok_l, corners_l = cv2.findChessboardCorners(gray_l, self.config.chessboard_size, None)
            ok_r, corners_r = cv2.findChessboardCorners(gray_r, self.config.chessboard_size, None)
            if not (ok_l and ok_r):
                continue
            corners_l = cv2.cornerSubPix(gray_l, corners_l, win, (-1, -1), self._criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, win, (-1, -1), self._criteria)
            objpoints.append(self._objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            if show:
                cv2.drawChessboardCorners(img_l, self.config.chessboard_size, corners_l, ok_l)
                cv2.drawChessboardCorners(img_r, self.config.chessboard_size, corners_r, ok_r)
                cv2.imshow("Left", img_l)
                cv2.imshow("Right", img_r)
                cv2.waitKey(500)
        if show:
            cv2.destroyAllWindows()

        if not objpoints:
            raise ValueError("No chessboard detected in any image pair")
        return objpoints, imgpoints_left, imgpoints_right, image_size

    def calibrate(self, show: bool = False, save: bool = True) -> StereoCalibrationResult:
        objpoints, imgpoints_l, imgpoints_r, image_size = self.find_corners(show=show)

        _, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, image_size, None, None)
        _, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None)

        rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            K1, D1, K2, D2, image_size,
            criteria=self._criteria, flags=cv2.CALIB_FIX_INTRINSIC,
        )
        _, _, _, _, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        result = StereoCalibrationResult(
            K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, Q=Q, rms=rms, image_size=image_size,
        )
        if save:
            result.save(self.config.output_path)
        return result
