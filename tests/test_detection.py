import numpy as np

from padelvision.config import DetectionConfig
from padelvision.detection.tracker import locate_ball


def test_locate_ball_finds_single_blob():
    import cv2

    heatmap = np.zeros((360, 640), dtype=np.uint8)
    cv2.circle(heatmap, (320, 180), 4, 255, -1)
    result = locate_ball(heatmap, DetectionConfig())
    assert result is not None
    x, y = result
    assert abs(x - 320) <= 5
    assert abs(y - 180) <= 5


def test_locate_ball_empty_heatmap_returns_none():
    heatmap = np.zeros((360, 640), dtype=np.uint8)
    assert locate_ball(heatmap, DetectionConfig()) is None
