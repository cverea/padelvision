from padelvision.calibration.capture import CameraCapture
from padelvision.config import CaptureConfig


def test_capture_photo_uses_injected_fetcher(tmp_path):
    calls = []

    def fake_fetch(url, timeout_s):
        calls.append(url)
        return b"jpeg-bytes"

    cfg = CaptureConfig(
        camera1_ip="1.2.3.4",
        camera1_dir=str(tmp_path / "cam1"),
        camera2_dir=str(tmp_path / "cam2"),
    )
    cap = CameraCapture(cfg, fetcher=fake_fetch)
    out = cap.capture_photo(cfg.camera1_ip, cfg.camera1_dir, 1)

    assert out.read_bytes() == b"jpeg-bytes"
    assert calls == ["http://1.2.3.4:8080/photo.jpg"]


def test_capture_pair_writes_both(tmp_path):
    cfg = CaptureConfig(
        camera1_dir=str(tmp_path / "cam1"),
        camera2_dir=str(tmp_path / "cam2"),
    )
    cap = CameraCapture(cfg, fetcher=lambda url, t: b"x")
    left, right = cap.capture_pair(3)
    assert left.exists() and right.exists()
    assert left.name == "photo3.jpg" and right.name == "photo3.jpg"
