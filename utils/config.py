"""
Configuration module for the AI Driver Drowsiness Detection System.

All thresholds, landmark indices, weights, and color constants are defined
here so they can be tuned without touching detector or model code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# ---------------------------------------------------------------------------

# 6-point eye landmarks (P1=outer corner, P2-P3=upper lid, P4=inner corner,
# P5-P6=lower lid) — same ordering used by the classic EAR formula.
LEFT_EYE_IDXS: List[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS: List[int] = [33, 160, 158, 133, 153, 144]

# Mouth landmarks used for MAR
MOUTH_OUTER_IDXS: List[int] = [61, 291, 39, 181, 0, 17, 269, 405]

# Simplified mouth corners + inner-lip top/bottom for fast MAR
MOUTH_LEFT: int = 61
MOUTH_RIGHT: int = 291
MOUTH_TOP_OUTER: int = 0
MOUTH_BOT_OUTER: int = 17
MOUTH_TOP_INNER: int = 13
MOUTH_BOT_INNER: int = 14

# 6 stable face points used for solvePnP head pose estimation
HEAD_POSE_IDXS: Dict[str, int] = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Corresponding 3-D model coordinates (mm, approximate generic face)
HEAD_POSE_3D: List[Tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),           # nose tip
    (0.0, -330.0, -65.0),      # chin
    (-225.0, 170.0, -135.0),   # left eye corner
    (225.0, 170.0, -135.0),    # right eye corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),   # right mouth corner
]


# ---------------------------------------------------------------------------
# Dataclass configs
# ---------------------------------------------------------------------------

@dataclass
class EARConfig:
    """Eye Aspect Ratio thresholds and smoothing."""
    # Eyes considered closed below this value
    threshold: float = 0.25
    # EMA alpha for smoothing (lower = smoother but more lag)
    smoothing_alpha: float = 0.4
    # Consecutive frames below threshold before counting as closed
    consec_frames: int = 2


@dataclass
class MARConfig:
    """Mouth Aspect Ratio thresholds and smoothing."""
    # Mouth considered open (yawn) above this value
    threshold: float = 0.55
    smoothing_alpha: float = 0.4
    consec_frames: int = 3


@dataclass
class HeadPoseConfig:
    """Head pose angle thresholds (degrees)."""
    pitch_threshold: float = 15.0    # nodding forward
    yaw_threshold: float = 25.0      # looking sideways
    roll_threshold: float = 20.0     # tilting head


@dataclass
class DrowsinessConfig:
    """Drowsiness scoring thresholds and component weights."""
    # Classification boundaries (score 0-1)
    alert_boundary: float = 0.35
    slight_boundary: float = 0.65

    # Eye closure durations (seconds) before increasing score
    closure_slight_secs: float = 1.5
    closure_drowsy_secs: float = 3.0

    # Blink rate boundaries (blinks per minute)
    blink_low: float = 8.0    # below = micro-sleeps
    blink_high: float = 30.0  # above = eye strain / fatigue

    # Feature weights (must sum to 1.0)
    w_ear: float = 0.40
    w_duration: float = 0.25
    w_yawn: float = 0.15
    w_headpose: float = 0.12
    w_blink: float = 0.08

    # State hysteresis: seconds in a state before transitioning
    hysteresis_enter: float = 1.0   # seconds to confirm worse state
    hysteresis_exit: float = 2.0    # seconds to confirm better state


@dataclass
class AppConfig:
    """Top-level application configuration."""
    ear: EARConfig = field(default_factory=EARConfig)
    mar: MARConfig = field(default_factory=MARConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)

    # MediaPipe detection confidence
    face_min_confidence: float = 0.70
    face_tracking_confidence: float = 0.50

    # Video
    frame_width: int = 640
    frame_height: int = 480

    # History / chart window
    history_size: int = 300          # number of frames kept
    chart_window_secs: int = 30      # seconds shown in trend charts

    # Screenshot / export
    screenshot_dir: str = "data/screenshots"
    log_csv_path: str = "data/detection_log.csv"

    # Display overlay flags
    show_landmarks: bool = True
    show_ear_gauge: bool = True
    show_fps: bool = True


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# BGR tuples for OpenCV overlays
STATUS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Alert": (50, 205, 50),
    "Slightly Drowsy": (0, 165, 255),
    "Drowsy": (0, 50, 220),
    "No Face": (160, 160, 160),
}

# Hex strings for Streamlit UI
STATUS_HEX: Dict[str, str] = {
    "Alert": "#32CD32",
    "Slightly Drowsy": "#FFA500",
    "Drowsy": "#DC143C",
    "No Face": "#888888",
}

# UI accent gradient
GRADIENT_PRIMARY = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
GRADIENT_ALERT = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
GRADIENT_SAFE = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"


def get_config() -> AppConfig:
    """Return default application configuration."""
    return AppConfig()
