"""
Metrics tracking module for the Driver Drowsiness Detection System.

Maintains rolling history of EAR, MAR, drowsiness scores, and derived
statistics. Also handles blink counting, CSV logging, and session reports.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """One frame's worth of detection data."""
    timestamp: float
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    drowsiness_score: float
    status: str
    confidence: float
    fps: float
    face_detected: bool


@dataclass
class SessionStats:
    """Aggregated statistics for the current session."""
    session_start: float = field(default_factory=time.time)
    total_frames: int = 0
    alert_frames: int = 0
    slight_frames: int = 0
    drowsy_frames: int = 0
    no_face_frames: int = 0
    total_blinks: int = 0
    total_yawns: int = 0
    peak_drowsiness_score: float = 0.0
    longest_closure_secs: float = 0.0

    @property
    def session_duration_secs(self) -> float:
        return time.time() - self.session_start

    @property
    def alert_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.alert_frames / self.total_frames * 100

    @property
    def slight_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.slight_frames / self.total_frames * 100

    @property
    def drowsy_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.drowsy_frames / self.total_frames * 100


# ---------------------------------------------------------------------------
# Blink detector helper
# ---------------------------------------------------------------------------

class BlinkCounter:
    """
    Detects blinks from a stream of EAR values.
    A blink is a dip below `threshold` that recovers above it.
    """

    def __init__(self, threshold: float = 0.25, consec_frames: int = 2) -> None:
        self.threshold = threshold
        self.consec_frames = consec_frames
        self._below_count: int = 0
        self._blink_count: int = 0
        self._in_blink: bool = False

    def update(self, ear: float) -> bool:
        """Feed one EAR value; returns True if a blink just completed."""
        blink_occurred = False
        if ear < self.threshold:
            self._below_count += 1
            self._in_blink = True
        else:
            if self._in_blink and self._below_count >= self.consec_frames:
                self._blink_count += 1
                blink_occurred = True
            self._below_count = 0
            self._in_blink = False
        return blink_occurred

    @property
    def total(self) -> int:
        return self._blink_count

    def reset(self) -> None:
        self._below_count = 0
        self._blink_count = 0
        self._in_blink = False


# ---------------------------------------------------------------------------
# Main metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Central store for all real-time and historical detection data.

    Maintains fixed-length deques for trend charts, computes rolling
    blink rates, and manages CSV logging.
    """

    def __init__(
        self,
        history_size: int = 300,
        log_csv_path: str = "data/detection_log.csv",
        ear_threshold: float = 0.25,
    ) -> None:
        self.history_size = history_size
        self.log_csv_path = log_csv_path

        # Rolling history deques
        self.timestamps: Deque[float] = deque(maxlen=history_size)
        self.ear_history: Deque[float] = deque(maxlen=history_size)
        self.mar_history: Deque[float] = deque(maxlen=history_size)
        self.score_history: Deque[float] = deque(maxlen=history_size)
        self.fps_history: Deque[float] = deque(maxlen=history_size)
        self.status_history: Deque[str] = deque(maxlen=history_size)
        self.pitch_history: Deque[float] = deque(maxlen=history_size)
        self.yaw_history: Deque[float] = deque(maxlen=history_size)

        # Blink tracking
        self.blink_counter = BlinkCounter(threshold=ear_threshold)
        self._blink_timestamps: Deque[float] = deque(maxlen=200)

        # Yawn tracking
        self._yawn_count: int = 0
        self._in_yawn: bool = False

        # Session-level stats
        self.stats = SessionStats()

        # Latest values (thread-safe reads from main thread)
        self.latest: Optional[FrameRecord] = None

        # CSV setup
        self._csv_initialized = False
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public update API                                                    #
    # ------------------------------------------------------------------ #

    def update(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        drowsiness_score: float,
        status: str,
        confidence: float,
        fps: float,
        face_detected: bool,
        mar_threshold: float = 0.55,
    ) -> FrameRecord:
        """Record one frame and return the FrameRecord."""
        now = time.time()

        # Blink
        blinked = self.blink_counter.update(ear) if face_detected else False
        if blinked:
            self._blink_timestamps.append(now)
            self.stats.total_blinks += 1

        # Yawn (rising edge of MAR threshold)
        if face_detected:
            if mar >= mar_threshold and not self._in_yawn:
                self._in_yawn = True
            elif mar < mar_threshold and self._in_yawn:
                self._in_yawn = False
                self._yawn_count += 1
                self.stats.total_yawns += 1

        # Append to rolling history
        self.timestamps.append(now)
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.score_history.append(drowsiness_score)
        self.fps_history.append(fps)
        self.status_history.append(status)
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)

        # Session stats
        self.stats.total_frames += 1
        if status == "Alert":
            self.stats.alert_frames += 1
        elif status == "Slightly Drowsy":
            self.stats.slight_frames += 1
        elif status == "Drowsy":
            self.stats.drowsy_frames += 1
        else:
            self.stats.no_face_frames += 1

        self.stats.peak_drowsiness_score = max(
            self.stats.peak_drowsiness_score, drowsiness_score
        )

        record = FrameRecord(
            timestamp=now,
            ear=ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            drowsiness_score=drowsiness_score,
            status=status,
            confidence=confidence,
            fps=fps,
            face_detected=face_detected,
        )
        self.latest = record

        # Async-safe CSV write
        self._log_to_csv(record)

        return record

    # ------------------------------------------------------------------ #
    # Derived statistics                                                   #
    # ------------------------------------------------------------------ #

    def blink_rate_per_minute(self, window_secs: float = 60.0) -> float:
        """Blinks per minute computed over the last `window_secs`."""
        now = time.time()
        cutoff = now - window_secs
        recent = sum(1 for t in self._blink_timestamps if t >= cutoff)
        elapsed = min(window_secs, now - self.stats.session_start)
        if elapsed < 1.0:
            return 0.0
        return recent / elapsed * 60.0

    def avg_ear(self, n: int = 30) -> float:
        """Mean EAR over last n frames."""
        arr = list(self.ear_history)[-n:]
        return float(np.mean(arr)) if arr else 0.0

    def avg_mar(self, n: int = 30) -> float:
        """Mean MAR over last n frames."""
        arr = list(self.mar_history)[-n:]
        return float(np.mean(arr)) if arr else 0.0

    def avg_fps(self, n: int = 30) -> float:
        """Mean FPS over last n frames."""
        arr = list(self.fps_history)[-n:]
        return float(np.mean(arr)) if arr else 0.0

    def score_trend(self) -> List[float]:
        """Return drowsiness score history as a plain list."""
        return list(self.score_history)

    def ear_trend(self) -> List[float]:
        return list(self.ear_history)

    def time_trend(self) -> List[float]:
        """Relative timestamps in seconds from session start."""
        start = self.stats.session_start
        return [t - start for t in self.timestamps]

    def yawn_count(self) -> int:
        return self._yawn_count

    def status_distribution(self) -> Dict[str, float]:
        """Percentage time in each state for the session."""
        return {
            "Alert": self.stats.alert_pct,
            "Slightly Drowsy": self.stats.slight_pct,
            "Drowsy": self.stats.drowsy_pct,
        }

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def export_csv(self, path: Optional[str] = None) -> str:
        """Write full session history to CSV and return the path."""
        out_path = path or self.log_csv_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rows = zip(
            self.time_trend(),
            list(self.ear_history),
            list(self.mar_history),
            list(self.score_history),
            list(self.status_history),
            list(self.fps_history),
        )
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "ear", "mar", "score", "status", "fps"])
            writer.writerows(rows)
        logger.info("Exported CSV to %s", out_path)
        return out_path

    def session_report(self) -> Dict:
        """Return a dict suitable for display or JSON export."""
        s = self.stats
        return {
            "session_duration_secs": round(s.session_duration_secs, 1),
            "total_frames": s.total_frames,
            "alert_pct": round(s.alert_pct, 1),
            "slight_pct": round(s.slight_pct, 1),
            "drowsy_pct": round(s.drowsy_pct, 1),
            "total_blinks": s.total_blinks,
            "total_yawns": s.total_yawns,
            "blink_rate_bpm": round(self.blink_rate_per_minute(), 1),
            "peak_score": round(s.peak_drowsiness_score, 3),
            "avg_fps": round(self.avg_fps(), 1),
        }

    # ------------------------------------------------------------------ #
    # Internal CSV logging                                                 #
    # ------------------------------------------------------------------ #

    def _log_to_csv(self, record: FrameRecord) -> None:
        """Append one record to the session CSV log."""
        try:
            write_header = not self._csv_initialized
            with open(self.log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        ["timestamp", "ear", "mar", "pitch", "yaw", "roll",
                         "score", "status", "confidence", "fps", "face_detected"]
                    )
                    self._csv_initialized = True
                writer.writerow([
                    datetime.fromtimestamp(record.timestamp).isoformat(),
                    round(record.ear, 4),
                    round(record.mar, 4),
                    round(record.pitch, 2),
                    round(record.yaw, 2),
                    round(record.roll, 2),
                    round(record.drowsiness_score, 4),
                    record.status,
                    round(record.confidence, 2),
                    round(record.fps, 1),
                    record.face_detected,
                ])
        except Exception as exc:  # noqa: BLE001
            logger.debug("CSV log error: %s", exc)
