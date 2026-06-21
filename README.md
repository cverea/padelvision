# PadelVision

Reconstruct the 3D trajectory of a padel ball from two synchronized camera
views. The pipeline is:

1. **Detect** the ball in each frame of each camera's video (TrackNet → heatmap → Hough circle).
2. **Calibrate** the stereo camera pair from chessboard image pairs.
3. **Triangulate** the matched 2D detections into 3D points.
4. **Visualize** the trajectory inside a model of a padel court.

## Install

```bash
pip install -e .                 # core (numpy, opencv-python, pandas)
pip install -e ".[detection]"    # + tensorflow/keras for ball detection
pip install -e ".[viz]"          # + matplotlib for plotting
pip install -e ".[capture]"      # + requests/keyboard for live capture
pip install -e ".[dev]"          # + pytest
```

## Usage

Everything is exposed through the `padelvision` CLI (or `python -m padelvision.cli`):

```bash
padelvision write-config config.json            # write a default config to edit
padelvision capture                             # grab calibration photos from two IP webcams
padelvision calibrate                           # stereo-calibrate -> stereo_params.yml
padelvision detect match.mp4 --stub balls.pkl   # detect the ball, cache to a stub
padelvision triangulate cam1.csv cam2.csv --params stereo_params.yml --plot
padelvision court                               # plot the court model
```

Pass `--config config.json` before the subcommand to override defaults.

### As a library

```python
from padelvision import Config
from padelvision.geometry import Triangulator

triangulator = Triangulator.from_stereo_params("stereo_params.yml")
points_3d = triangulator.triangulate(cam1_xy, cam2_xy)  # (N,2),(N,2) -> (N,3)
```

## Layout

- `padelvision/config.py` — all tunable parameters as dataclasses (JSON-loadable).
- `padelvision/detection/` — TrackNet model + `BallTracker`.
- `padelvision/calibration/` — `StereoCalibrator` and IP-webcam `CameraCapture`.
- `padelvision/geometry/` — court model + `Triangulator`.
- `padelvision/viz/` — matplotlib plotting (no import-time side effects).
- `padelvision/cli.py` — command-line entry point.
- `tests/` — pytest suite for the pure logic.

Data lives at the repo root: `weights/` (model weights), `camera1/`+`camera2/`
(calibration image pairs), `stereo_params.yml` (calibration output),
`test_left.jpg`/`test_right.jpg`.

## Tests

```bash
pytest
```
