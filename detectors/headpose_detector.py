"""
Head pose detector module — estimates pitch, yaw, and roll from face landmarks.

Uses OpenCV's solvePnP with 6 stable facial landmarks to fit a 3-D head
model, then decomposes the rotation matrix into Euler angles.

Pitch  → nodding (positive = chin down)
Yaw    → turning left/right (positive = face turned right)
Roll   → tilting left/right (positive = tilt right)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3-D face model (generic, in mm)
# ---------------------------------------------------------------------------

# These 6 points correspond to landmark indices in HEAD_POSE_IDXS below.
FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],  # nose tip         (idx 1)
    [0.0,  -330.0,  -65.0  ],  # chin             (idx 152)
    [-225.0, 170.0, -135.0 ],  # left eye corner  (idx 263)
    [225.0,  170.0, -135.0 ],  # right eye corner (idx 33)
    [-150.0,-150.0, -125.0 ],  # left mouth       (idx 61)
    [150.0, -150.0, -125.0 ],  # right mouth      (idx 291)
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
HEAD_POSE_LM_IDXS: List[int] = [1, 152, 263, 33, 61, 291]


# ---------------------------------------------------------------------------
# HeadPoseDetector class
# ---------------------------------------------------------------------------

class HeadPoseDetector:
    """
    Estimates head orientation (pitch, yaw, roll) in degrees via solvePnP.

    Smooths the Euler angles with EMA to reduce jitter.
    """

    def __init__(
        self,
        pitch_threshold: float = 15.0,
        yaw_threshold: float = 25.0,
        roll_threshold: float = 20.0,
        smoothing_alpha: float = 0.35,
    ) -> None:
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.roll_threshold = roll_threshold
        self.alpha = smoothing_alpha

        # Smoothed angle state
        self._pitch: float = 0.0
        self._yaw: float = 0.0
        self._roll: float = 0.0

        # Cached camera matrix (updated when frame size changes)
        self._cam_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._last_frame_shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    # Core processing                                                      #
    # ------------------------------------------------------------------ #

    def process(
        self,
        face_landmarks,
        frame_shape: Tuple[int, int],
    ) -> dict:
        """
        Estimate head pose angles.

        Returns:
            dict with keys: pitch, yaw, roll (degrees), is_nodding,
            is_looking_away, nose_2d, nose_3d_projected
        """
        h, w = frame_shape[:2]
        self._update_camera_matrix(h, w)

        lm = face_landmarks.landmark
        img_points = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in HEAD_POSE_LM_IDXS],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_POINTS,
            img_points,
            self._cam_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return self._get_current_result()

        # Decompose rotation vector → Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        pitch_r, yaw_r, roll_r = self._rotation_matrix_to_euler(rmat)

        pitch_deg = math.degrees(pitch_r)
        yaw_deg = math.degrees(yaw_r)
        roll_deg = math.degrees(roll_r)

        # EMA smoothing
        self._pitch = self.alpha * pitch_deg + (1.0 - self.alpha) * self._pitch
        self._yaw = self.alpha * yaw_deg + (1.0 - self.alpha) * self._yaw
        self._roll = self.alpha * roll_deg + (1.0 - self.alpha) * self._roll

        # Project nose tip for drawing the pose axis line
        nose_2d = (int(lm[1].x * w), int(lm[1].y * h))
        nose_3d_projected, _ = cv2.projectPoints(
            np.array([[0.0, 0.0, 1000.0]]),
            rvec, tvec,
            self._cam_matrix,
            self._dist_coeffs,
        )
        nose_end = (
            int(nose_3d_projected[0][0][0]),
            int(nose_3d_projected[0][0][1]),
        )

        return {
            "pitch": round(self._pitch, 2),
            "yaw": round(self._yaw, 2),
            "roll": round(self._roll, 2),
            "is_nodding": abs(self._pitch) > self.pitch_threshold,
            "is_looking_away": abs(self._yaw) > self.yaw_threshold,
            "is_tilted": abs(self._roll) > self.roll_threshold,
            "nose_2d": nose_2d,
            "nose_end": nose_end,
        }

    def get_no_face_result(self) -> dict:
        """Return zeroed result when no face is detected."""
        return self._get_current_result(no_face=True)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _update_camera_matrix(self, h: int, w: int) -> None:
        """Rebuild camera matrix only when frame dimensions change."""
        if self._last_frame_shape == (h, w):
            return
        focal = w  # approximate: focal length ≈ frame width in pixels
        center = (w / 2.0, h / 2.0)
        self._cam_matrix = np.array([
            [focal, 0,      center[0]],
            [0,     focal,  center[1]],
            [0,     0,      1        ],
        ], dtype=np.float64)
        self._last_frame_shape = (h, w)

    @staticmethod
    def _rotation_matrix_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a 3×3 rotation matrix to Euler angles (pitch, yaw, roll).

        Uses the ZYX convention (yaw → pitch → roll).
        Handles gimbal lock gracefully.
        """
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw   = math.atan2(-rmat[2, 0], sy)
            roll  = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw   = math.atan2(-rmat[2, 0], sy)
            roll  = 0.0

        return pitch, yaw, roll

    def _get_current_result(self, no_face: bool = False) -> dict:
        return {
            "pitch": 0.0 if no_face else round(self._pitch, 2),
            "yaw": 0.0 if no_face else round(self._yaw, 2),
            "roll": 0.0 if no_face else round(self._roll, 2),
            "is_nodding": abs(self._pitch) > self.pitch_threshold,
            "is_looking_away": abs(self._yaw) > self.yaw_threshold,
            "is_tilted": abs(self._roll) > self.roll_threshold,
            "nose_2d": None,
            "nose_end": None,
        }

    def reset(self) -> None:
        """Reset smoothed angles."""
        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
