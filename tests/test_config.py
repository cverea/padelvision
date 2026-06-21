import json

from padelvision.config import CaptureConfig, Config


def test_defaults_are_populated():
    cfg = Config()
    assert cfg.calibration.chessboard_size == (7, 7)
    assert cfg.court.length == 20.0
    assert cfg.detection.n_classes == 256


def test_photo_url():
    cap = CaptureConfig(camera1_ip="10.0.0.5", port=8080, photo_endpoint="photo.jpg")
    assert cap.photo_url("10.0.0.5") == "http://10.0.0.5:8080/photo.jpg"


def test_round_trip_json(tmp_path):
    cfg = Config()
    cfg.court.length = 21.5
    cfg.detection.hough_param2 = 3.0
    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.court.length == 21.5
    assert loaded.detection.hough_param2 == 3.0
    assert isinstance(loaded.capture, CaptureConfig)


def test_from_dict_ignores_unknown_keys():
    data = {"court": {"length": 19.0, "bogus": 1}, "unknown_section": {}}
    cfg = Config.from_dict(data)
    assert cfg.court.length == 19.0
    # Untouched sections keep their defaults.
    assert cfg.detection.n_classes == 256


def test_to_dict_is_json_serializable():
    json.dumps(Config().to_dict())  # should not raise
