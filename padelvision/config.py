"""Central configuration for the PadelVision pipeline.

All tunable parameters live here as plain dataclasses so they can be created with
sensible defaults, overridden in code, or loaded from / saved to a JSON file.
The module intentionally depends only on the standard library so it can be
imported without OpenCV, TensorFlow, etc.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Tuple, get_type_hints


@dataclass
class CaptureConfig:
    """Settings for grabbing calibration photos from two IP webcams."""

    camera1_ip: str = "192.168.0.13"
    camera2_ip: str = "192.168.0.25"
    port: int = 8080
    photo_endpoint: str = "photo.jpg"
    video_endpoint: str = "video"  # IP Webcam MJPEG stream path
    camera1_dir: str = "camera1"
    camera2_dir: str = "camera2"
    camera1_video: str = "camera1.avi"  # recorded video output paths
    camera2_video: str = "camera2.avi"
    record_fps: float = 30.0
    capture_key: str = "p"
    quit_key: str = "q"
    timeout_s: float = 5.0

    def photo_url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}/{self.photo_endpoint}"

    def video_url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}/{self.video_endpoint}"


@dataclass
class CalibrationConfig:
    """Settings for stereo calibration from chessboard image pairs."""

    chessboard_cols: int = 7
    chessboard_rows: int = 7
    square_size: float = 1.0  # real-world size of a square; scales translation units
    left_dir: str = "camera1"
    right_dir: str = "camera2"
    image_glob: str = "*.jpg"
    output_path: str = "stereo_params.yml"
    # Sub-pixel corner refinement / stereo termination criteria.
    criteria_max_iter: int = 30
    criteria_epsilon: float = 1e-3
    corner_window: int = 11

    @property
    def chessboard_size(self) -> Tuple[int, int]:
        return (self.chessboard_cols, self.chessboard_rows)


@dataclass
class DetectionConfig:
    """Settings for TrackNet ball detection."""

    weights_path: str = "weights/model.3"
    n_classes: int = 256
    width: int = 640
    height: int = 360
    # Heatmap post-processing.
    binary_threshold: int = 127
    # cv2.HoughCircles parameters.
    hough_dp: float = 1.0
    hough_min_dist: float = 1.0
    hough_param1: float = 50.0
    hough_param2: float = 2.0
    hough_min_radius: int = 2
    hough_max_radius: int = 7


@dataclass
class CourtConfig:
    """Physical dimensions of a padel court, in metres."""

    length: float = 20.0
    width: float = 10.0
    height_wall: float = 3.0
    height_fence: float = 4.0
    service_line_offset: float = 3.0  # distance of the service line from each back wall
    distance_from_back_wall: float = 4.0  # where the side fence meets the wall height


@dataclass
class Config:
    """Aggregate configuration for the whole pipeline."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    court: CourtConfig = field(default_factory=CourtConfig)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _from_dict(cls, data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _from_dict(klass: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys are ignored so old/extra config files stay loadable.
    """
    if not is_dataclass(klass):
        return data
    kwargs: dict[str, Any] = {}
    # Resolve string annotations (caused by `from __future__ import annotations`)
    # back into real types so nested dataclasses are detected.
    hints = get_type_hints(klass)
    field_names = {f.name for f in fields(klass)}
    for name, value in data.items():
        if name not in field_names:
            continue
        field_type = hints.get(name)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _from_dict(field_type, value)
        else:
            kwargs[name] = value
    return klass(**kwargs)
