"""
Yawn detector module — computes Mouth Aspect Ratio (MAR) and detects yawns.

MAR is analogous to EAR: it measures how far the mouth is open relative to
its width.  A sustained high MAR value indicates a yawn.

MAR = vertical_opening / horizontal_width
    = dist(top_inner, bot_inner) / dist(left_corner, right_corner)

An extended version averages three vertical measurements for robustness.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# YawnDetector class
# ---------------------------------------------------------------------------

class YawnDetector:
    """
    Stateful yawn detector using Mouth Aspect Ratio (MAR).

    Landmark indices (MediaPipe Face Mesh):
        Horizontal: 61 (left corner) → 291 (right corner)
        Vertical top-outer: 0   (cupid's bow centre)
        Vertical top-inner: 13  (inner upper lip)
        Vertical bot-inner: 14  (inner lower lip)
        Vertical bot-outer: 17  (lower lip centre)
    """

    MOUTH_LEFT: int = 61
    MOUTH_RIGHT: int = 291
    MOUTH_TOP_OUTER: int = 0
    MOUTH_TOP_INNER: int = 13
    MOUTH_BOT_INNER: int = 14
    MOUTH_BOT_OUTER: int = 17

    # Additional landmarks for drawing
    MOUTH_DRAW_IDXS: List[int] = [
        61, 291, 39, 181, 0, 17, 269, 405,
        78, 308, 82, 87, 13, 14, 312, 317,
    ]

    def __init__(
        self,
        mar_threshold: float = 0.55,
        smoothing_alpha: float = 0.35,
        consec_frames: int = 3,
        min_yawn_duration_secs: float = 0.5,
    ) -> None:
        """
        Args:
            mar_threshold: MAR above which mouth is considered yawning.
            smoothing_alpha: EMA factor (lower = smoother, more lag).
            consec_frames: Frames above threshold before confirming yawn.
            min_yawn_duration_secs: Minimum open duration to count as yawn.
        """
        self.mar_threshold = mar_threshold
        self.alpha = smoothing_alpha
        self.consec_frames = consec_frames
        self.min_yawn_duration = min_yawn_duration_secs

        # Smoothing state
        self._smooth_mar: float = 0.1

        # Yawn state machine
        self._above_count: int = 0
        self._in_yawn: bool = False
        self._yawn_start: Optional[float] = None
        self._yawn_duration: float = 0.0

        # Counters
        self._yawn_count: int = 0
        self._yawn_times: Deque[float] = deque(maxlen=100)

    # ------------------------------------------------------------------ #
    # Core processing                                                      #
    # ------------------------------------------------------------------ #

    def process(
        self,
        face_landmarks,
        frame_shape: Tuple[int, int],
    ) -> dict:
        """
        Compute MAR and update yawn state.

        Args:
            face_landmarks: MediaPipe NormalizedLandmarkList
            frame_shape: (height, width)

        Returns:
            dict with keys:
                mar, is_yawning, yawn_duration_secs, yawn_count,
                mouth_coords (pixel list for drawing)
        """
        h, w = frame_shape[:2]
        lm = face_landmarks.landmark
        coords = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in range(len(lm))],
            dtype=np.float32,
        )

        raw_mar = self._compute_mar(coords)

        # EMA smoothing
        self._smooth_mar = (
            self.alpha * raw_mar + (1.0 - self.alpha) * self._smooth_mar
        )

        # State update
        self._update_state(self._smooth_mar)

        return {
            "mar": round(self._smooth_mar, 4),
            "raw_mar": round(raw_mar, 4),
            "is_yawning": self._in_yawn,
            "yawn_duration_secs": round(self._yawn_duration, 2),
            "yawn_count": self._yawn_count,
            "mouth_coords": [
                (int(coords[i][0]), int(coords[i][1]))
                for i in self.MOUTH_DRAW_IDXS
                if i < len(coords)
            ],
        }

    def get_no_face_result(self) -> dict:
        """Return neutral result when no face detected."""
        self._reset_yawn_timing()
        return {
            "mar": 0.1,
            "raw_mar": 0.1,
            "is_yawning": False,
            "yawn_duration_secs": 0.0,
            "yawn_count": self._yawn_count,
            "mouth_coords": [],
        }

    # ------------------------------------------------------------------ #
    # MAR calculation                                                      #
    # ------------------------------------------------------------------ #

    def _compute_mar(self, coords: np.ndarray) -> float:
        """
        MAR = mean(vertical_openings) / horizontal_width.

        Uses three vertical samples for robustness against asymmetric opening.
        """
        try:
            left = coords[self.MOUTH_LEFT]
            right = coords[self.MOUTH_RIGHT]
            top_outer = coords[self.MOUTH_TOP_OUTER]
            top_inner = coords[self.MOUTH_TOP_INNER]
            bot_inner = coords[self.MOUTH_BOT_INNER]
            bot_outer = coords[self.MOUTH_BOT_OUTER]

            horiz = _dist(left, right)
            if horiz < 1e-6:
                return 0.0

            v1 = _dist(top_outer, bot_outer)
            v2 = _dist(top_inner, bot_inner)
            v3 = (v1 + v2) / 2.0  # blended measure

            return (v1 + v2 + v3) / (3.0 * horiz)
        except (IndexError, ValueError) as exc:
            logger.debug("MAR computation error: %s", exc)
            return 0.0

    # ------------------------------------------------------------------ #
    # Yawn state machine                                                   #
    # ------------------------------------------------------------------ #

    def _update_state(self, mar: float) -> None:
        now = time.time()

        if mar >= self.mar_threshold:
            self._above_count += 1
            if self._above_count >= self.consec_frames and not self._in_yawn:
                self._in_yawn = True
                self._yawn_start = now
            if self._yawn_start is not None:
                self._yawn_duration = now - self._yawn_start
        else:
            if self._in_yawn:
                duration = now - (self._yawn_start or now)
                if duration >= self.min_yawn_duration:
                    self._yawn_count += 1
                    self._yawn_times.append(now)
            self._in_yawn = False
            self._above_count = 0
            self._reset_yawn_timing()

    def _reset_yawn_timing(self) -> None:
        self._yawn_start = None
        self._yawn_duration = 0.0

    def yawn_rate_per_hour(self, window_secs: float = 600.0) -> float:
        """Yawns per hour over the last `window_secs`."""
        now = time.time()
        cutoff = now - window_secs
        recent = sum(1 for t in self._yawn_times if t >= cutoff)
        elapsed = min(window_secs, now - (self._yawn_times[0] if self._yawn_times else now))
        if elapsed < 1.0:
            return 0.0
        return recent / elapsed * 3600.0

    def reset(self) -> None:
        """Reset all state."""
        self._smooth_mar = 0.1
        self._above_count = 0
        self._in_yawn = False
        self._yawn_start = None
        self._yawn_duration = 0.0
        self._yawn_count = 0
        self._yawn_times.clear()
