"""
Eye detector module — computes Eye Aspect Ratio (EAR) and detects blinks.

Uses MediaPipe Face Mesh normalized landmarks; no model training required.

EAR formula (Soukupová & Čech, 2016):
    EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)

where p1..p6 are the six eye landmarks in the standard order:
    p1 = outer corner, p2 = upper-outer, p3 = upper-inner,
    p4 = inner corner, p5 = lower-inner, p6 = lower-outer
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(landmarks_2d: np.ndarray, idxs: List[int]) -> float:
    """
    Compute EAR for one eye given a [N×2] array of (x, y) pixel coords
    and the 6-landmark index list for that eye.

    Returns 0.0 if any landmark is missing or computation fails.
    """
    try:
        p1 = landmarks_2d[idxs[0]]
        p2 = landmarks_2d[idxs[1]]
        p3 = landmarks_2d[idxs[2]]
        p4 = landmarks_2d[idxs[3]]
        p5 = landmarks_2d[idxs[4]]
        p6 = landmarks_2d[idxs[5]]

        vertical_1 = _dist(p2, p6)
        vertical_2 = _dist(p3, p5)
        horizontal = _dist(p1, p4)

        if horizontal < 1e-6:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    except (IndexError, ValueError) as exc:
        logger.debug("EAR computation error: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# EyeDetector class
# ---------------------------------------------------------------------------

class EyeDetector:
    """
    Stateful eye detector that tracks EAR, blinks, and eye-closure duration.

    Usage (per frame):
        result = detector.process(face_landmarks, frame_shape)
    """

    # Landmark indices (MediaPipe Face Mesh canonical order)
    LEFT_EYE_IDXS: List[int] = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDXS: List[int] = [33, 160, 158, 133, 153, 144]

    # Pupils for drawing
    LEFT_PUPIL_IDX: int = 468
    RIGHT_PUPIL_IDX: int = 473

    def __init__(
        self,
        ear_threshold: float = 0.25,
        smoothing_alpha: float = 0.4,
        consec_frames: int = 2,
    ) -> None:
        """
        Args:
            ear_threshold: EAR below which eye is considered closed.
            smoothing_alpha: EMA factor for EAR smoothing (0 < α ≤ 1).
            consec_frames: Minimum frames below threshold to count as blink.
        """
        self.ear_threshold = ear_threshold
        self.alpha = smoothing_alpha
        self.consec_frames = consec_frames

        # EAR smoothing state
        self._smooth_ear: float = 0.3

        # Closure tracking
        self._below_count: int = 0
        self._closure_start: Optional[float] = None
        self._closure_duration: float = 0.0
        self._longest_closure: float = 0.0

        # Blink counting
        self._blink_count: int = 0
        self._in_blink: bool = False
        self._blink_times: Deque[float] = deque(maxlen=200)

        # Last known values (returned when no face detected)
        self._last_left_ear: float = 0.3
        self._last_right_ear: float = 0.3

    # ------------------------------------------------------------------ #
    # Core processing                                                      #
    # ------------------------------------------------------------------ #

    def process(
        self,
        face_landmarks,
        frame_shape: Tuple[int, int],
    ) -> dict:
        """
        Compute EAR and update blink/closure state.

        Args:
            face_landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            frame_shape: (height, width) of the original frame.

        Returns:
            dict with keys:
                left_ear, right_ear, mean_ear,
                is_closed, closure_duration_secs,
                blink_count, blink_rate_bpm,
                left_coords, right_coords   (pixel coords for drawing)
        """
        h, w = frame_shape[:2]

        # Build pixel-coordinate array [N×2]
        lm = face_landmarks.landmark
        coords = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in range(len(lm))],
            dtype=np.float32,
        )

        left_ear = _eye_aspect_ratio(coords, self.LEFT_EYE_IDXS)
        right_ear = _eye_aspect_ratio(coords, self.RIGHT_EYE_IDXS)
        raw_mean = (left_ear + right_ear) / 2.0

        # Exponential moving average smoothing
        self._smooth_ear = (
            self.alpha * raw_mean + (1.0 - self.alpha) * self._smooth_ear
        )

        self._last_left_ear = left_ear
        self._last_right_ear = right_ear

        # Update closure / blink state
        self._update_state(self._smooth_ear)

        return {
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "mean_ear": round(self._smooth_ear, 4),
            "is_closed": self._smooth_ear < self.ear_threshold,
            "closure_duration_secs": round(self._closure_duration, 2),
            "longest_closure_secs": round(self._longest_closure, 2),
            "blink_count": self._blink_count,
            "blink_rate_bpm": round(self._blink_rate_bpm(), 1),
            # Pixel coords for drawing overlays
            "left_coords": [
                (int(coords[i][0]), int(coords[i][1]))
                for i in self.LEFT_EYE_IDXS
            ],
            "right_coords": [
                (int(coords[i][0]), int(coords[i][1]))
                for i in self.RIGHT_EYE_IDXS
            ],
        }

    def get_no_face_result(self) -> dict:
        """Return neutral result when no face is detected."""
        self._reset_closure()
        return {
            "left_ear": 0.3,
            "right_ear": 0.3,
            "mean_ear": 0.3,
            "is_closed": False,
            "closure_duration_secs": 0.0,
            "longest_closure_secs": self._longest_closure,
            "blink_count": self._blink_count,
            "blink_rate_bpm": round(self._blink_rate_bpm(), 1),
            "left_coords": [],
            "right_coords": [],
        }

    # ------------------------------------------------------------------ #
    # Internal state machine                                               #
    # ------------------------------------------------------------------ #

    def _update_state(self, ear: float) -> None:
        now = time.time()

        if ear < self.ear_threshold:
            self._below_count += 1
            if not self._in_blink:
                self._in_blink = True
            # Start timing closure
            if self._closure_start is None:
                self._closure_start = now
            self._closure_duration = now - self._closure_start
            self._longest_closure = max(
                self._longest_closure, self._closure_duration
            )
        else:
            if self._in_blink and self._below_count >= self.consec_frames:
                self._blink_count += 1
                self._blink_times.append(now)
            self._in_blink = False
            self._below_count = 0
            self._reset_closure()

    def _reset_closure(self) -> None:
        self._closure_start = None
        self._closure_duration = 0.0

    def _blink_rate_bpm(self, window_secs: float = 60.0) -> float:
        now = time.time()
        cutoff = now - window_secs
        recent = sum(1 for t in self._blink_times if t >= cutoff)
        elapsed = min(window_secs, now - (self._blink_times[0] if self._blink_times else now))
        if elapsed < 1.0:
            return 0.0
        return recent / elapsed * 60.0

    def reset(self) -> None:
        """Reset all state (call on session restart)."""
        self._smooth_ear = 0.3
        self._below_count = 0
        self._closure_start = None
        self._closure_duration = 0.0
        self._longest_closure = 0.0
        self._blink_count = 0
        self._in_blink = False
        self._blink_times.clear()
