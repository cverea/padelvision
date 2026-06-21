"""Capture synchronized photos from two IP webcams (e.g. the "IP Webcam" app).

``requests`` and ``keyboard`` are imported lazily so this module can be imported
(and ``capture_photo`` unit-tested with a fake fetcher) without those packages
or a keyboard device being available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from padelvision.config import CaptureConfig


def _default_fetcher(url: str, timeout_s: float) -> bytes:
    import requests  # lazy: only needed for real captures

    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    return response.content


class CameraCapture:
    def __init__(
        self,
        config: CaptureConfig | None = None,
        fetcher: Optional[Callable[[str, float], bytes]] = None,
    ):
        self.config = config or CaptureConfig()
        # Inject a custom fetcher in tests; defaults to a real HTTP GET.
        self._fetch = fetcher or _default_fetcher
        self.photo_number = 1

    def capture_photo(self, ip: str, folder: str, photo_number: int) -> Path:
        """Fetch one photo and save it as ``photo{n}.jpg`` in ``folder``."""
        Path(folder).mkdir(parents=True, exist_ok=True)
        url = self.config.photo_url(ip)
        content = self._fetch(url, self.config.timeout_s)
        out = Path(folder) / f"photo{photo_number}.jpg"
        out.write_bytes(content)
        return out

    def capture_pair(self, photo_number: int) -> tuple[Path, Path]:
        """Capture one synchronized image from each camera."""
        left = self.capture_photo(self.config.camera1_ip, self.config.camera1_dir, photo_number)
        right = self.capture_photo(self.config.camera2_ip, self.config.camera2_dir, photo_number)
        return left, right

    def run(self) -> None:
        """Interactive loop: press the capture key to grab a pair, quit key to stop."""
        import keyboard  # lazy: only needed for the interactive loop
        import time

        print(
            f"Press '{self.config.capture_key}' to capture from both cameras. "
            f"Press '{self.config.quit_key}' to quit."
        )
        while True:
            if keyboard.is_pressed(self.config.capture_key):
                print(f"Capturing pair {self.photo_number}...")
                try:
                    self.capture_pair(self.photo_number)
                    self.photo_number += 1
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    print(f"Capture failed: {exc}")
                time.sleep(1)
            if keyboard.is_pressed(self.config.quit_key):
                print("Exiting.")
                break
