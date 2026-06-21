"""Command-line entry point for the PadelVision pipeline.

Run ``python -m padelvision.cli <command> --help`` (or the installed
``padelvision`` script) for details. Heavy dependencies are imported inside each
command so unrelated commands stay fast and importable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from padelvision.config import Config


def _load_config(path: str | None) -> Config:
    return Config.load(path) if path else Config()


def _cmd_capture(args: argparse.Namespace) -> None:
    from padelvision.calibration import CameraCapture

    CameraCapture(_load_config(args.config).capture).run()


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from padelvision.calibration import StereoCalibrator

    result = StereoCalibrator(_load_config(args.config).calibration).calibrate(
        show=args.show, save=True
    )
    print(f"Stereo calibration RMS error: {result.rms:.4f}")
    print(f"Saved parameters to {_load_config(args.config).calibration.output_path}")


def _cmd_record(args: argparse.Namespace) -> None:
    from padelvision.calibration import CameraCapture

    capture = CameraCapture(_load_config(args.config).capture)
    out1, out2 = capture.record_videos(duration_s=args.duration)
    print(f"Recorded {out1} and {out2}")


def _cmd_detect(args: argparse.Namespace) -> None:
    from padelvision.detection import BallTracker

    detections = BallTracker(_load_config(args.config).detection).detect(
        args.video, read_from_stub=args.read_from_stub, stub_path=args.stub
    )
    found = sum(1 for v in detections.values() if v is not None)
    print(f"Processed {len(detections)} frames, ball found in {found}.")


def _cmd_triangulate(args: argparse.Namespace) -> None:
    import numpy as np

    from padelvision.geometry import Triangulator

    pts1 = np.loadtxt(args.cam1, delimiter=",").reshape(-1, 2)
    pts2 = np.loadtxt(args.cam2, delimiter=",").reshape(-1, 2)
    triangulator = Triangulator.from_stereo_params(args.params)
    points_3d = triangulator.triangulate(pts1, pts2)
    if args.out:
        np.savetxt(args.out, points_3d, delimiter=",")
        print(f"Wrote {len(points_3d)} 3D points to {args.out}")
    else:
        print(points_3d)
    if args.plot:
        from padelvision.viz import plot_court_with_trajectory

        plot_court_with_trajectory(points_3d, _load_config(args.config).court)


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    import numpy as np

    from padelvision.detection import BallTracker
    from padelvision.geometry import Triangulator, matched_points

    cfg = _load_config(args.config)
    tracker = BallTracker(cfg.detection)  # one model, reused for both views
    det1 = tracker.detect(args.video1, read_from_stub=args.read_from_stub, stub_path=args.stub1)
    det2 = tracker.detect(args.video2, read_from_stub=args.read_from_stub, stub_path=args.stub2)

    pts1, pts2, frames = matched_points(det1, det2)
    print(f"Ball found in both views on {len(frames)} frames.")
    if not frames:
        return

    points_3d = Triangulator.from_stereo_params(args.params).triangulate(pts1, pts2)
    if args.out:
        np.savetxt(args.out, points_3d, delimiter=",")
        print(f"Wrote {len(points_3d)} 3D points to {args.out}")
    else:
        print(points_3d)
    if args.plot:
        from padelvision.viz import plot_court_with_trajectory

        plot_court_with_trajectory(points_3d, cfg.court)


def _cmd_court(args: argparse.Namespace) -> None:
    from padelvision.viz import plot_court

    plot_court(_load_config(args.config).court)


def _cmd_write_config(args: argparse.Namespace) -> None:
    Config().save(args.path)
    print(f"Wrote default config to {args.path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="padelvision", description=__doc__)
    parser.add_argument("--config", help="Path to a JSON config file (optional).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capture", help="Capture calibration photos from two IP webcams.")
    p.set_defaults(func=_cmd_capture)

    p = sub.add_parser("calibrate", help="Stereo-calibrate from chessboard image pairs.")
    p.add_argument("--show", action="store_true", help="Display detected corners.")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("record", help="Record two camera videos simultaneously.")
    p.add_argument(
        "--duration", type=float,
        help="Seconds to record (default: until the quit key is pressed).",
    )
    p.set_defaults(func=_cmd_record)

    p = sub.add_parser("detect", help="Detect the ball in a video.")
    p.add_argument("video", help="Path to the input video.")
    p.add_argument("--stub", help="Path to read/write cached detections.")
    p.add_argument("--read-from-stub", action="store_true", dest="read_from_stub")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("triangulate", help="Triangulate 2D detections into 3D points.")
    p.add_argument("cam1", help="CSV of camera-1 (x,y) points.")
    p.add_argument("cam2", help="CSV of camera-2 (x,y) points.")
    p.add_argument("--params", default="stereo_params.yml", help="Stereo params YAML.")
    p.add_argument("--out", help="CSV path to write 3D points (otherwise printed).")
    p.add_argument("--plot", action="store_true", help="Plot the trajectory on the court.")
    p.set_defaults(func=_cmd_triangulate)

    p = sub.add_parser(
        "reconstruct",
        help="Detect the ball in two videos and triangulate its 3D trajectory.",
    )
    p.add_argument("video1", help="Path to the camera-1 video.")
    p.add_argument("video2", help="Path to the camera-2 video.")
    p.add_argument("--params", default="stereo_params.yml", help="Stereo params YAML.")
    p.add_argument("--stub1", help="Cache path for camera-1 detections.")
    p.add_argument("--stub2", help="Cache path for camera-2 detections.")
    p.add_argument("--read-from-stub", action="store_true", dest="read_from_stub")
    p.add_argument("--out", help="CSV path to write 3D points (otherwise printed).")
    p.add_argument("--plot", action="store_true", help="Plot the trajectory on the court.")
    p.set_defaults(func=_cmd_reconstruct)

    p = sub.add_parser("court", help="Plot the padel court model.")
    p.set_defaults(func=_cmd_court)

    p = sub.add_parser("write-config", help="Write a default JSON config file.")
    p.add_argument("path", help="Output path, e.g. config.json.")
    p.set_defaults(func=_cmd_write_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
