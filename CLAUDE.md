# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PadelVision reconstructs the 3D trajectory of a padel ball from two synchronized camera views. The pipeline is: **detect** the ball in each video frame (2D) → stereo-**calibrate** the camera pair → **triangulate** the 2D detections into 3D points → **visualize** the trajectory inside a model of a padel court.

The code is organized as a single installable `padelvision` package; every stage is config-driven, importable without side effects, and exposed through a CLI.

## Architecture

- `padelvision/config.py` — **all tunable parameters** live here as dataclasses (`CaptureConfig`, `CalibrationConfig`, `DetectionConfig`, `CourtConfig`, aggregated by `Config`). Stdlib-only (no cv2/numpy) so it imports anywhere; `Config.save/load` round-trips JSON and `from_dict` ignores unknown keys. When adding a parameter, add it here rather than hard-coding it in a module.
- `padelvision/detection/` — `tracknet.build_tracknet()` (Keras imported lazily inside the function so the package imports without TensorFlow); `tracker.BallTracker` runs the model over a video. `tracker.locate_ball(heatmap, config)` is the pure heatmap→circle step, separated so it's unit-testable without Keras.
- `padelvision/calibration/` — `stereo.StereoCalibrator` (returns a `StereoCalibrationResult` dataclass and optionally writes `stereo_params.yml`; GUI display is opt-in via `show=`). `capture.CameraCapture` pulls frames from two IP webcams; its HTTP fetch is injectable (`fetcher=`) so it's testable, and `requests`/`keyboard` are imported lazily.
- `padelvision/geometry/` — `court.court_segments()/court_keypoints()` build the court purely (no plotting). `triangulation.Triangulator` does the real triangulation; `Triangulator.from_stereo_params("stereo_params.yml")` builds projection matrices `P1 = K1[I|0]`, `P2 = K2[R|T]` from the calibration file.
- `padelvision/viz/plotting.py` — matplotlib plots (`plot_court`, `plot_trajectory`, `plot_court_with_trajectory`). Lazy matplotlib import, **nothing runs at import time**.
- `padelvision/cli.py` — argparse entry point (`padelvision <command>` or `python -m padelvision.cli`); each command imports its heavy deps lazily.

Key design rules to preserve: **no work at import time**, **heavy deps (keras/matplotlib/requests/keyboard) imported lazily**, and **parameters come from `config.py`**.

## Commands

```bash
pip install -e ".[dev]"     # core + pytest (add [detection]/[viz]/[capture] as needed)
pytest                      # run the test suite
pytest tests/test_triangulation.py::test_triangulate_recovers_known_points  # single test

padelvision write-config config.json   # emit default config to edit
padelvision calibrate [--show]         # stereo-calibrate camera1/+camera2/ -> stereo_params.yml
padelvision detect <video> --stub balls.pkl
padelvision triangulate cam1.csv cam2.csv --params stereo_params.yml --plot
padelvision court
```

When installing with `--no-deps` (deps already present), opencv may be the `opencv-python-headless` build — fine for tests, but GUI calls (`cv2.imshow`) won't display.

## Dependencies

Declared in `pyproject.toml`: core = numpy, opencv-python, pandas; optional extras `detection` (tensorflow, keras), `viz` (matplotlib), `capture` (requests, keyboard), `dev` (pytest). TrackNet uses `data_format='channels_first'` and 3 stacked frames (9 channels, 360×640 input).

## Data files (repo root)

`weights/` (model weights), `camera1/`+`camera2/` (chessboard pairs — must be index-aligned and equal in count), `stereo_params.yml` (calibration output), `test_left.jpg`/`test_right.jpg`. Calibration/triangulation resolve these relative to the working directory, so run from the repo root.

## Tests

`tests/` covers the pure logic that runs without TensorFlow: config round-trip, court geometry, triangulation (projects known 3D points and recovers them within tolerance), `stereo_params.yml` I/O, `locate_ball` on a synthetic heatmap, and capture with an injected fetcher. Detection-via-Keras and live capture are intentionally not unit-tested (no model weights / hardware in CI).
