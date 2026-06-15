"""
Drowsiness classifier module.

Combines EAR, MAR, head-pose, and blink-rate signals into a single
drowsiness score (0–1) and classifies it into three states:

    Alert          score < 0.35
    Slightly Drowsy  0.35 ≤ score < 0.65
    Drowsy          score ≥ 0.65

A hysteresis state machine prevents rapid toggling between states.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from utils.config import (
    STATUS_BGR, STATUS_HEX,
    DrowsinessConfig, EARConfig, MARConfig, HeadPoseConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DrowsinessResult:
    """Full output of one classifier evaluation."""
    status: str           # "Alert" | "Slightly Drowsy" | "Drowsy" | "No Face"
    score: float          # 0.0–1.0
    confidence: float     # 0–100 %
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    is_closed: bool
    closure_secs: float
    is_yawning: bool
    blink_rate: float
    color_bgr: tuple
    color_hex: str
    component_scores: Dict[str, float]


# ---------------------------------------------------------------------------
# Drowsiness classifier
# ---------------------------------------------------------------------------

class DrowsinessClassifier:
    """
    Rule-based multi-feature drowsiness classifier.

    Scoring weights (configurable via DrowsinessConfig):
        EAR component       : 40 %
        Closure duration    : 25 %
        Yawning (MAR)       : 15 %
        Head pose           : 12 %
        Blink frequency     :  8 %
    """

    STATES = ["Alert", "Slightly Drowsy", "Drowsy"]

    def __init__(
        self,
        ear_cfg: Optional[EARConfig] = None,
        mar_cfg: Optional[MARConfig] = None,
        pose_cfg: Optional[HeadPoseConfig] = None,
        drowsy_cfg: Optional[DrowsinessConfig] = None,
    ) -> None:
        self.ear_cfg = ear_cfg or EARConfig()
        self.mar_cfg = mar_cfg or MARConfig()
        self.pose_cfg = pose_cfg or HeadPoseConfig()
        self.cfg = drowsy_cfg or DrowsinessConfig()

        # State machine
        self._current_state: str = "Alert"
        self._candidate_state: str = "Alert"
        self._candidate_since: float = time.time()

        # Score smoothing (EMA)
        self._smooth_score: float = 0.0
        self._score_alpha: float = 0.3

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        eye_result: dict,
        yawn_result: dict,
        pose_result: dict,
    ) -> DrowsinessResult:
        """
        Combine detector outputs into a single classified result.

        Args:
            eye_result:  output of EyeDetector.process()
            yawn_result: output of YawnDetector.process()
            pose_result: output of HeadPoseDetector.process()

        Returns:
            DrowsinessResult with status, score, confidence, and raw values.
        """
        ear = eye_result.get("mean_ear", 0.3)
        closure = eye_result.get("closure_duration_secs", 0.0)
        blink_rate = eye_result.get("blink_rate_bpm", 15.0)
        is_closed = eye_result.get("is_closed", False)

        mar = yawn_result.get("mar", 0.1)
        is_yawning = yawn_result.get("is_yawning", False)

        pitch = pose_result.get("pitch", 0.0)
        yaw = pose_result.get("yaw", 0.0)
        roll = pose_result.get("roll", 0.0)
        is_nodding = pose_result.get("is_nodding", False)
        is_looking_away = pose_result.get("is_looking_away", False)

        # Compute component scores
        ear_score = self._score_ear(ear)
        dur_score = self._score_closure(closure)
        yawn_score = self._score_yawn(mar, is_yawning)
        pose_score = self._score_pose(pitch, yaw, is_nodding, is_looking_away)
        blink_score = self._score_blink(blink_rate)

        raw_score = (
            self.cfg.w_ear      * ear_score   +
            self.cfg.w_duration * dur_score   +
            self.cfg.w_yawn     * yawn_score  +
            self.cfg.w_headpose * pose_score  +
            self.cfg.w_blink    * blink_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        # EMA smoothing
        self._smooth_score = (
            self._score_alpha * raw_score +
            (1.0 - self._score_alpha) * self._smooth_score
        )
        score = self._smooth_score

        # Hysteresis state machine
        status = self._update_state(score)
        confidence = self._compute_confidence(score, status)

        return DrowsinessResult(
            status=status,
            score=round(score, 4),
            confidence=round(confidence, 1),
            ear=round(ear, 4),
            mar=round(mar, 4),
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            is_closed=is_closed,
            closure_secs=round(closure, 2),
            is_yawning=is_yawning,
            blink_rate=round(blink_rate, 1),
            color_bgr=STATUS_BGR.get(status, (150, 150, 150)),
            color_hex=STATUS_HEX.get(status, "#888888"),
            component_scores={
                "EAR": round(ear_score, 3),
                "Closure": round(dur_score, 3),
                "Yawn": round(yawn_score, 3),
                "Head Pose": round(pose_score, 3),
                "Blink Rate": round(blink_score, 3),
            },
        )

    def no_face_result(self) -> DrowsinessResult:
        """Return a safe result when no face is detected."""
        self._smooth_score = max(0.0, self._smooth_score - 0.02)
        return DrowsinessResult(
            status="No Face",
            score=0.0,
            confidence=0.0,
            ear=0.0, mar=0.0,
            pitch=0.0, yaw=0.0, roll=0.0,
            is_closed=False, closure_secs=0.0,
            is_yawning=False, blink_rate=0.0,
            color_bgr=STATUS_BGR["No Face"],
            color_hex=STATUS_HEX["No Face"],
            component_scores={},
        )

    def reset(self) -> None:
        self._current_state = "Alert"
        self._candidate_state = "Alert"
        self._candidate_since = time.time()
        self._smooth_score = 0.0

    # ------------------------------------------------------------------ #
    # Component scoring functions (each returns 0.0–1.0)                  #
    # ------------------------------------------------------------------ #

    def _score_ear(self, ear: float) -> float:
        """Higher score when EAR is low (eyes closing)."""
        thr = self.ear_cfg.threshold
        if ear >= thr:
            return 0.0
        # Normalise: EAR=0 → score=1, EAR=threshold → score=0
        return float(np.clip((thr - ear) / thr * 1.5, 0.0, 1.0))

    def _score_closure(self, duration_secs: float) -> float:
        """Ramp from 0 at 0s closure to 1 at drowsy-threshold closure."""
        if duration_secs <= 0.0:
            return 0.0
        max_dur = self.cfg.closure_drowsy_secs
        return float(np.clip(duration_secs / max_dur, 0.0, 1.0))

    def _score_yawn(self, mar: float, is_yawning: bool) -> float:
        """Higher score when MAR exceeds threshold (mouth wide open)."""
        thr = self.mar_cfg.threshold
        if mar <= thr:
            return 0.0
        base = float(np.clip((mar - thr) / thr, 0.0, 1.0))
        return base * (1.2 if is_yawning else 0.7)

    def _score_pose(
        self, pitch: float, yaw: float,
        is_nodding: bool, is_looking_away: bool
    ) -> float:
        """Penalise extreme pitch (nodding) and yaw (distraction)."""
        p_thr = self.pose_cfg.pitch_threshold
        y_thr = self.pose_cfg.yaw_threshold

        pitch_score = float(np.clip((abs(pitch) - p_thr) / p_thr, 0.0, 1.0)) \
            if abs(pitch) > p_thr else 0.0
        yaw_score = float(np.clip((abs(yaw) - y_thr) / y_thr * 0.6, 0.0, 1.0)) \
            if abs(yaw) > y_thr else 0.0

        return float(np.clip(pitch_score * 0.7 + yaw_score * 0.3, 0.0, 1.0))

    def _score_blink(self, bpm: float) -> float:
        """
        Anomalous blink rate → fatigue signal.
        Very low bpm (micro-sleeps) or very high bpm (eye strain) score high.
        """
        if bpm <= 0:
            return 0.0
        low = self.cfg.blink_low
        high = self.cfg.blink_high
        if bpm < low:
            return float(np.clip(1.0 - bpm / low, 0.0, 1.0))
        if bpm > high:
            return float(np.clip((bpm - high) / high * 0.5, 0.0, 1.0))
        return 0.0

    # ------------------------------------------------------------------ #
    # Hysteresis state machine                                             #
    # ------------------------------------------------------------------ #

    def _update_state(self, score: float) -> str:
        """
        Update the state machine with hysteresis to avoid rapid toggling.

        Requires the candidate state to persist for:
            - `hysteresis_enter` seconds to move to a worse state
            - `hysteresis_exit`  seconds to move to a better state
        """
        if score < self.cfg.alert_boundary:
            candidate = "Alert"
        elif score < self.cfg.slight_boundary:
            candidate = "Slightly Drowsy"
        else:
            candidate = "Drowsy"

        now = time.time()
        if candidate != self._candidate_state:
            self._candidate_state = candidate
            self._candidate_since = now

        elapsed = now - self._candidate_since
        state_idx = self.STATES.index(self._current_state)
        cand_idx = self.STATES.index(self._candidate_state)

        if cand_idx > state_idx:
            # Moving to worse state — short wait
            if elapsed >= self.cfg.hysteresis_enter:
                self._current_state = self._candidate_state
        elif cand_idx < state_idx:
            # Recovering to better state — longer wait
            if elapsed >= self.cfg.hysteresis_exit:
                self._current_state = self._candidate_state

        return self._current_state

    # ------------------------------------------------------------------ #
    # Confidence computation                                               #
    # ------------------------------------------------------------------ #

    def _compute_confidence(self, score: float, status: str) -> float:
        """
        Map the raw score to a 0–100 % confidence value for the
        predicted state.
        """
        if status == "Alert":
            # Confidence = how far below the alert boundary
            return (1.0 - score / self.cfg.alert_boundary) * 100.0
        elif status == "Drowsy":
            # Confidence = how far above the slight boundary
            span = 1.0 - self.cfg.slight_boundary
            return min(100.0, (score - self.cfg.slight_boundary) / span * 100.0)
        else:  # Slightly Drowsy
            centre = (self.cfg.alert_boundary + self.cfg.slight_boundary) / 2.0
            dist = 1.0 - abs(score - centre) / centre
            return float(np.clip(dist * 100.0, 50.0, 100.0))


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def draw_status_overlay(
    frame: np.ndarray,
    result: DrowsinessResult,
    fps: float,
    show_landmarks: bool = True,
    eye_coords: Optional[tuple] = None,
    mouth_coords: Optional[list] = None,
    nose_2d: Optional[tuple] = None,
    nose_end: Optional[tuple] = None,
) -> np.ndarray:
    """
    Draw diagnostic overlays onto the frame in-place and return it.

    Overlays:
        - Status banner (top)
        - EAR / MAR / Score values (top-left)
        - FPS counter (top-right)
        - Eye and mouth landmark dots
        - Head-pose direction arrow
    """
    h, w = frame.shape[:2]
    color = result.color_bgr

    # ── Status banner ──────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    status_text = f"{result.status}  {result.confidence:.0f}%"
    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    tx = (w - tw) // 2
    cv2.putText(frame, status_text, (tx, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Metric labels (bottom-left) ─────────────────────────────────────
    metrics = [
        f"EAR:   {result.ear:.3f}",
        f"MAR:   {result.mar:.3f}",
        f"Score: {result.score:.3f}",
        f"Blink: {result.blink_rate:.0f} bpm",
        f"Pitch: {result.pitch:.1f}°  Yaw: {result.yaw:.1f}°",
    ]
    y0 = h - 10 - len(metrics) * 22
    for i, line in enumerate(metrics):
        y = y0 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── FPS (top-right) ─────────────────────────────────────────────────
    fps_text = f"FPS: {fps:.0f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, fps_text, (w - fw - 10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Eye landmarks ────────────────────────────────────────────────────
    if show_landmarks and eye_coords:
        l_coords, r_coords = eye_coords
        for (x, y) in l_coords:
            cv2.circle(frame, (x, y), 2, (0, 230, 0), -1)
        for (x, y) in r_coords:
            cv2.circle(frame, (x, y), 2, (0, 230, 0), -1)

    # ── Mouth landmarks ───────────────────────────────────────────────────
    if show_landmarks and mouth_coords:
        mouth_color = (0, 165, 255) if result.is_yawning else (0, 200, 200)
        for (x, y) in mouth_coords[:8]:
            cv2.circle(frame, (x, y), 2, mouth_color, -1)

    # ── Head-pose arrow ───────────────────────────────────────────────────
    if nose_2d and nose_end:
        cv2.arrowedLine(frame, nose_2d, nose_end,
                        (200, 100, 255), 2, tipLength=0.2)

    # ── Closure progress bar ─────────────────────────────────────────────
    if result.is_closed and result.closure_secs > 0:
        max_secs = 3.0
        frac = min(1.0, result.closure_secs / max_secs)
        bar_w = int(w * frac)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), color, -1)

    return frame
