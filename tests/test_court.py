import numpy as np

from padelvision.config import CourtConfig
from padelvision.geometry.court import court_keypoints, court_segments


def test_segments_are_3d_point_pairs():
    segments = court_segments()
    assert len(segments) > 0
    for start, end in segments:
        assert start.shape == (3,)
        assert end.shape == (3,)


def test_keypoints_within_court_bounds():
    cfg = CourtConfig()
    pts = court_keypoints(cfg)
    assert pts.shape[1] == 3
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= cfg.length
    assert pts[:, 1].min() >= 0 and pts[:, 1].max() <= cfg.width
    assert pts[:, 2].min() >= 0 and pts[:, 2].max() <= cfg.height_fence


def test_floor_corners_present():
    cfg = CourtConfig()
    pts = court_keypoints(cfg)
    for corner in ([0, 0, 0], [cfg.length, 0, 0], [cfg.length, cfg.width, 0], [0, cfg.width, 0]):
        assert np.any(np.all(np.isclose(pts, corner), axis=1)), corner


def test_dimensions_scale_with_config():
    big = court_keypoints(CourtConfig(length=40, width=20))
    assert big[:, 0].max() == 40
    assert big[:, 1].max() == 20
