"""Geometric model of a padel court.

Pure NumPy geometry — no plotting and no import-time side effects. Use
``padelvision.viz.plotting`` to render the segments returned here.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from padelvision.config import CourtConfig

# A line segment is a (start, end) pair of 3D points.
Segment = Tuple[np.ndarray, np.ndarray]


def court_segments(config: CourtConfig | None = None) -> List[Segment]:
    """Return the list of 3D line segments that draw a padel court.

    Coordinates are in metres with the origin at one floor corner: X along the
    court length, Y across the width, Z up.
    """
    c = config or CourtConfig()
    length, width = c.length, c.width
    wall, fence = c.height_wall, c.height_fence
    s = c.service_line_offset
    d = c.distance_from_back_wall

    def seg(p0, p1) -> Segment:
        return (np.array(p0, dtype=float), np.array(p1, dtype=float))

    segments: List[Segment] = []

    # Floor outline.
    outline = [
        [0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0], [0, 0, 0],
    ]
    for a, b in zip(outline, outline[1:]):
        segments.append(seg(a, b))

    # Service lines (parallel to the net) and the centre service line.
    segments.append(seg([s, 0, 0], [s, width, 0]))
    segments.append(seg([length - s, 0, 0], [length - s, width, 0]))
    segments.append(seg([s, width / 2, 0], [length - s, width / 2, 0]))

    # Net line.
    segments.append(seg([length / 2, 0, 0], [length / 2, width, 0]))

    # Back walls (vertical posts + top rail) at both ends.
    for x in (0, length):
        segments.append(seg([x, 0, 0], [x, 0, wall]))
        segments.append(seg([x, width, 0], [x, width, wall]))
        segments.append(seg([x, 0, wall], [x, width, wall]))

    # Side walls (top rail at wall height) along both sides.
    for y in (0, width):
        segments.append(seg([0, y, wall], [length, y, wall]))

    # Fence top rails at the back ends.
    for x in (0, length):
        segments.append(seg([x, 0, fence], [x, width, fence]))

    # Side fence rails near the back walls and their vertical connectors.
    for y in (0, width):
        segments.append(seg([0, y, fence], [d, y, fence]))
        segments.append(seg([length - d, y, fence], [length, y, fence]))
    for x in (0, length):
        segments.append(seg([x, 0, wall], [x, 0, fence]))
        segments.append(seg([x, width, wall], [x, width, fence]))

    return segments


def court_keypoints(config: CourtConfig | None = None) -> np.ndarray:
    """Return the unique 3D keypoints of the court as an (N, 3) array."""
    segments = court_segments(config)
    points = np.vstack([np.array([a, b]) for a, b in segments])
    return np.unique(points, axis=0)
