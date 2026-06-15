"""
AI Driver Drowsiness Detection System
======================================
Main Streamlit application.

Architecture:
    WebRTC video stream → MediaPipe Face Mesh → EyeDetector + YawnDetector
    + HeadPoseDetector → DrowsinessClassifier → Annotated frame + Metrics

Real-time video is handled by streamlit-webrtc (WebRTC in browser).
Detection results are passed thread-safely to the main Streamlit thread via
a shared threading.Lock + latest-result slot, then a st_autorefresh loop
reads and renders them.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from detectors.eye_detector import EyeDetector
from detectors.headpose_detector import HeadPoseDetector
from detectors.yawn_detector import YawnDetector
from models.drowsiness_model import DrowsinessClassifier, DrowsinessResult, draw_status_overlay
from styles.custom_css import (
    inject_css,
    metric_card_html,
    progress_bar_html,
    section_header_html,
    status_badge_html,
)
from utils.audio_manager import AudioManager
from utils.config import AppConfig, get_config
from utils.metrics import MetricsTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# MediaPipe initializer (cached resource — one instance per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_face_mesh(min_detection_confidence: float, min_tracking_confidence: float):
    """Load and return a MediaPipe FaceMesh instance."""
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


# ---------------------------------------------------------------------------
# Thread-safe result container
# ---------------------------------------------------------------------------
class DetectionResult:
    """Shared mutable slot for the latest detection result from the processor thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Optional[dict] = None

    def set(self, data: dict) -> None:
        with self._lock:
            self._data = data

    def get(self) -> Optional[dict]:
        with self._lock:
            return self._data


# ---------------------------------------------------------------------------
# Video Processor (runs in WebRTC thread)
# ---------------------------------------------------------------------------
class DrowsinessVideoProcessor:
    """
    Processes each video frame:
        1. Runs MediaPipe Face Mesh
        2. Calls EyeDetector, YawnDetector, HeadPoseDetector
        3. Runs DrowsinessClassifier
        4. Draws overlays
        5. Deposits latest result into shared DetectionResult slot

    Accessed via ctx.video_processor from the main thread.
    """

    def __init__(self, cfg: AppConfig, result_slot: DetectionResult) -> None:
        self.cfg = cfg
        self.result_slot = result_slot
        self.show_landmarks = cfg.show_landmarks

        # Detectors
        self.eye_detector = EyeDetector(
            ear_threshold=cfg.ear.threshold,
            smoothing_alpha=cfg.ear.smoothing_alpha,
            consec_frames=cfg.ear.consec_frames,
        )
        self.yawn_detector = YawnDetector(
            mar_threshold=cfg.mar.threshold,
            smoothing_alpha=cfg.mar.smoothing_alpha,
            consec_frames=cfg.mar.consec_frames,
        )
        self.headpose_detector = HeadPoseDetector(
            pitch_threshold=cfg.head_pose.pitch_threshold,
            yaw_threshold=cfg.head_pose.yaw_threshold,
            roll_threshold=cfg.head_pose.roll_threshold,
        )
        self.classifier = DrowsinessClassifier(
            ear_cfg=cfg.ear,
            mar_cfg=cfg.mar,
            pose_cfg=cfg.head_pose,
            drowsy_cfg=cfg.drowsiness,
        )

        # MediaPipe (loaded once per session)
        self.face_mesh = load_face_mesh(
            cfg.face_min_confidence, cfg.face_tracking_confidence
        )

        # FPS tracking
        self._prev_time: float = time.time()
        self._fps: float = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Called by streamlit-webrtc for each incoming video frame."""
        img = frame.to_ndarray(format="bgr24")
        processed, result_dict = self._process_frame(img)
        self.result_slot.set(result_dict)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run full detection pipeline on a single BGR frame."""
        now = time.time()
        dt = now - self._prev_time
        self._fps = 1.0 / dt if dt > 0 else 30.0
        self._prev_time = now

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            eye_r = self.eye_detector.process(landmarks, (h, w))
            yawn_r = self.yawn_detector.process(landmarks, (h, w))
            pose_r = self.headpose_detector.process(landmarks, (h, w))
            dr: DrowsinessResult = self.classifier.evaluate(eye_r, yawn_r, pose_r)

            annotated = draw_status_overlay(
                frame.copy(),
                dr,
                self._fps,
                show_landmarks=self.show_landmarks,
                eye_coords=(eye_r["left_coords"], eye_r["right_coords"]),
                mouth_coords=yawn_r["mouth_coords"],
                nose_2d=pose_r.get("nose_2d"),
                nose_end=pose_r.get("nose_end"),
            )

            result_dict = {
                "face_detected": True,
                "status": dr.status,
                "score": dr.score,
                "confidence": dr.confidence,
                "ear": dr.ear,
                "mar": dr.mar,
                "pitch": dr.pitch,
                "yaw": dr.yaw,
                "roll": dr.roll,
                "is_closed": dr.is_closed,
                "closure_secs": dr.closure_secs,
                "is_yawning": dr.is_yawning,
                "blink_rate": dr.blink_rate,
                "blink_count": eye_r["blink_count"],
                "yawn_count": yawn_r["yawn_count"],
                "color_hex": dr.color_hex,
                "fps": round(self._fps, 1),
                "component_scores": dr.component_scores,
                "ts": now,
            }
        else:
            # No face detected
            dr = self.classifier.no_face_result()
            annotated = frame.copy()
            # Draw "No Face" label
            cv2.putText(
                annotated, "No Face Detected",
                (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (100, 100, 120), 2, cv2.LINE_AA,
            )
            result_dict = {
                "face_detected": False,
                "status": "No Face",
                "score": 0.0,
                "confidence": 0.0,
                "ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "is_closed": False,
                "closure_secs": 0.0,
                "is_yawning": False,
                "blink_rate": 0.0,
                "blink_count": 0,
                "yawn_count": 0,
                "color_hex": "#888888",
                "fps": round(self._fps, 1),
                "component_scores": {},
                "ts": now,
            }

        return annotated, result_dict


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state(cfg: AppConfig) -> None:
    """Ensure all session-state keys exist on first run."""
    defaults = {
        "cfg": cfg,
        "result_slot": DetectionResult(),
        "metrics": MetricsTracker(
            history_size=cfg.history_size,
            log_csv_path=cfg.log_csv_path,
            ear_threshold=cfg.ear.threshold,
        ),
        "audio_manager": AudioManager(),
        "alarm_enabled": True,
        "alarm_volume": 0.75,
        "show_landmarks": cfg.show_landmarks,
        "camera_index": 0,
        "session_started": time.time(),
        "screenshot_count": 0,
        "last_result": None,
        # Score and status for history charts
        "score_history": [],
        "ear_history": [],
        "time_history": [],
        "status_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Chart builders (Plotly)
# ---------------------------------------------------------------------------

def build_score_chart(time_vals: list, score_vals: list, status_vals: list) -> go.Figure:
    """Line chart of drowsiness score over time with coloured region fills."""
    fig = go.Figure()

    # Reference lines
    for threshold, label, color in [
        (0.35, "Alert boundary", "rgba(50,205,50,0.3)"),
        (0.65, "Drowsy boundary", "rgba(220,20,60,0.3)"),
    ]:
        fig.add_hline(
            y=threshold, line_dash="dot",
            line_color=color, opacity=0.6,
            annotation_text=label,
            annotation_font_color=color,
            annotation_font_size=10,
        )

    # Score trace
    fig.add_trace(go.Scatter(
        x=time_vals, y=score_vals,
        mode="lines",
        name="Drowsiness Score",
        line=dict(color="#667eea", width=2),
        fill="tozeroy",
        fillcolor="rgba(102,126,234,0.08)",
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#9090b0"),
        xaxis=dict(
            title="Time (s)", showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Score", range=[0, 1], showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(size=10),
        ),
        margin=dict(l=40, r=20, t=20, b=40),
        height=200,
        showlegend=False,
    )
    return fig


def build_ear_chart(time_vals: list, ear_vals: list, threshold: float) -> go.Figure:
    """Line chart of EAR over time."""
    fig = go.Figure()
    fig.add_hline(
        y=threshold, line_dash="dot",
        line_color="rgba(255,165,0,0.5)", opacity=0.7,
        annotation_text="Closed threshold",
        annotation_font_color="rgba(255,165,0,0.8)",
        annotation_font_size=10,
    )
    fig.add_trace(go.Scatter(
        x=time_vals, y=ear_vals,
        mode="lines",
        name="EAR",
        line=dict(color="#4facfe", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,172,254,0.08)",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#9090b0"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
        yaxis=dict(title="EAR", range=[0, 0.5], showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=20, b=30),
        height=180,
        showlegend=False,
    )
    return fig


def build_status_donut(alert_pct: float, slight_pct: float, drowsy_pct: float) -> go.Figure:
    """Donut chart showing time in each state."""
    labels = ["Alert", "Slightly Drowsy", "Drowsy"]
    values = [max(0.01, alert_pct), max(0.01, slight_pct), max(0.01, drowsy_pct)]
    colors = ["#32CD32", "#FFA500", "#DC143C"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#9090b0"),
        showlegend=True,
        legend=dict(
            font=dict(size=10, color="#9090b0"),
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
            x=1.0, y=0.5,
        ),
        margin=dict(l=0, r=80, t=10, b=10),
        height=170,
        annotations=[dict(
            text=f"{alert_pct:.0f}%<br><span style='font-size:9px'>Alert</span>",
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(color="#32CD32"),
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(cfg: AppConfig) -> dict:
    """Render all sidebar controls and return a dict of current settings."""
    settings = {}

    with st.sidebar:
        # ── Logo / Title ───────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:8px 0 16px">
            <div style="font-size:2.2rem">🚗</div>
            <div style="font-size:1.0rem;font-weight:800;background:linear-gradient(135deg,#667eea,#f093fb);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text">DrowseSafe AI</div>
            <div style="font-size:0.65rem;color:#5a5a7a;margin-top:2px">Driver Safety System</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        # ── Camera ─────────────────────────────────────────────────────
        st.markdown("**📷 Camera**")
        camera_idx = st.selectbox(
            "Camera source",
            options=[0, 1, 2],
            format_func=lambda x: f"Camera {x}" + (" (default)" if x == 0 else ""),
            key="cam_select",
        )
        settings["camera_index"] = camera_idx
        settings["show_landmarks"] = st.toggle(
            "Show facial landmarks", value=True, key="tog_landmarks"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Detection Thresholds ────────────────────────────────────────
        with st.expander("⚙️ Detection Thresholds", expanded=False):
            ear_thr = st.slider(
                "EAR closed threshold", 0.15, 0.40, cfg.ear.threshold, 0.01,
                help="Eye Aspect Ratio below this = eyes closed",
            )
            mar_thr = st.slider(
                "MAR yawn threshold", 0.40, 0.80, cfg.mar.threshold, 0.01,
                help="Mouth Aspect Ratio above this = yawning",
            )
            pitch_thr = st.slider(
                "Head pitch threshold (°)", 5.0, 40.0, cfg.head_pose.pitch_threshold, 1.0,
            )
            yaw_thr = st.slider(
                "Head yaw threshold (°)", 10.0, 50.0, cfg.head_pose.yaw_threshold, 1.0,
            )
            settings["ear_threshold"] = ear_thr
            settings["mar_threshold"] = mar_thr
            settings["pitch_threshold"] = pitch_thr
            settings["yaw_threshold"] = yaw_thr

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Alarm Settings ─────────────────────────────────────────────
        st.markdown("**🔔 Alarm Settings**")
        alarm_enabled = st.toggle("Enable alarm", value=True, key="alarm_toggle")
        volume = st.slider("Volume", 0.0, 1.0, 0.75, 0.05, key="alarm_vol")
        settings["alarm_enabled"] = alarm_enabled
        settings["alarm_volume"] = volume

        uploaded_alarm = st.file_uploader(
            "Custom alarm audio", type=["mp3", "wav", "ogg"], key="alarm_uploader",
            help="Upload your own alarm sound (MP3 / WAV / OGG)",
        )
        if uploaded_alarm is not None:
            audio_mgr: AudioManager = st.session_state.audio_manager
            mime = (
                "audio/mpeg" if uploaded_alarm.name.endswith(".mp3")
                else "audio/ogg" if uploaded_alarm.name.endswith(".ogg")
                else "audio/wav"
            )
            audio_mgr.load_from_bytes(uploaded_alarm.read(), mime)
            st.success("✓ Custom alarm loaded")

        # Test alarm button
        st.session_state.audio_manager.render_test_button_component(volume)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Export / Session ────────────────────────────────────────────
        st.markdown("**📊 Session**")
        metrics: MetricsTracker = st.session_state.metrics
        report = metrics.session_report()

        col_a, col_b = st.columns(2)
        with col_a:
            elapsed = int(report["session_duration_secs"])
            st.metric("Duration", f"{elapsed//60:02d}:{elapsed%60:02d}")
        with col_b:
            st.metric("Blinks", str(report["total_blinks"]))

        if st.button("📥 Export CSV", use_container_width=True):
            csv_path = metrics.export_csv()
            st.success(f"Saved: {csv_path}")

        if st.button("📋 Download Report", use_container_width=True):
            report_json = json.dumps(report, indent=2)
            st.download_button(
                "⬇ report.json", report_json,
                file_name=f"drowsiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.metrics = MetricsTracker(
                history_size=cfg.history_size,
                log_csv_path=cfg.log_csv_path,
                ear_threshold=cfg.ear.threshold,
            )
            st.session_state.score_history = []
            st.session_state.ear_history = []
            st.session_state.time_history = []
            st.session_state.status_log = []
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.6rem;color:#3a3a5a;text-align:center'>"
            "DrowseSafe AI v1.0 · MediaPipe Face Mesh<br>"
            "EAR + MAR + Head Pose Detection</div>",
            unsafe_allow_html=True,
        )

    return settings


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def render_main(settings: dict, cfg: AppConfig) -> None:
    """Build the full dashboard layout."""

    # ── App header ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="ddd-header">
        <span style="font-size:2.4rem">🚗</span>
        <div>
            <div class="ddd-header-title">AI Driver Drowsiness Detection</div>
            <div class="ddd-header-subtitle">
                Real-time Eye • Yawn • Head-Pose Analysis &nbsp;·&nbsp;
                MediaPipe Face Mesh &nbsp;·&nbsp; Production Ready
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── WebRTC + Status panel ────────────────────────────────────────────
    col_video, col_status = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown(
            section_header_html("Live Camera Feed", "📹"),
            unsafe_allow_html=True,
        )

        # Build RTC configuration with public STUN servers for cloud deploy
        rtc_config = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        }

        ctx = webrtc_streamer(
            key="drowsiness_detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=lambda: DrowsinessVideoProcessor(
                cfg=st.session_state.cfg,
                result_slot=st.session_state.result_slot,
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30, "max": 30},
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.state.playing:
            st.markdown(
                "<div style='font-size:0.72rem;color:#32CD32;margin-top:6px'>"
                "🟢 &nbsp;Camera active — detection running</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("👆 Click **START** to begin monitoring", icon="ℹ️")

        # Screenshot button
        if ctx.state.playing:
            if st.button("📸 Save Screenshot", use_container_width=True):
                _save_screenshot()

    with col_status:
        _render_status_panel()

    # ── Row 2: Eye metrics | Yawn metrics | Head pose ───────────────────
    st.markdown("---")
    col_eye, col_yawn, col_pose = st.columns(3, gap="medium")

    with col_eye:
        _render_eye_panel()
    with col_yawn:
        _render_yawn_panel()
    with col_pose:
        _render_headpose_panel()

    # ── Row 3: Charts ────────────────────────────────────────────────────
    st.markdown("---")
    tab_score, tab_ear, tab_dist, tab_log = st.tabs([
        "📈 Drowsiness Score", "👁 EAR Trend", "🥧 State Distribution", "📋 Event Log",
    ])

    m: MetricsTracker = st.session_state.metrics
    t_vals = st.session_state.time_history
    s_vals = st.session_state.score_history
    e_vals = st.session_state.ear_history
    stat_vals = st.session_state.status_log

    with tab_score:
        if len(s_vals) > 5:
            st.plotly_chart(
                build_score_chart(t_vals[-300:], s_vals[-300:], stat_vals[-300:]),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.caption("Start the camera to see the trend.")

    with tab_ear:
        if len(e_vals) > 5:
            st.plotly_chart(
                build_ear_chart(t_vals[-300:], e_vals[-300:], cfg.ear.threshold),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.caption("Start the camera to see the EAR trend.")

    with tab_dist:
        report = m.session_report()
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            fig = build_status_donut(
                report["alert_pct"], report["slight_pct"], report["drowsy_pct"]
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col_d2:
            st.markdown("<br>", unsafe_allow_html=True)
            for label, key, color in [
                ("Alert", "alert_pct", "#32CD32"),
                ("Slightly Drowsy", "slight_pct", "#FFA500"),
                ("Drowsy", "drowsy_pct", "#DC143C"),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;margin:4px 0'>"
                    f"<span style='font-size:0.8rem;color:{color}'>{label}</span>"
                    f"<span style='font-size:0.8rem;font-weight:700;color:{color}'>"
                    f"{report[key]:.1f}%</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Blink Rate", f"{report['blink_rate_bpm']} bpm")
            st.metric("Yawns", str(report["total_yawns"]))
            st.metric("Peak Score", f"{report['peak_score']:.3f}")

    with tab_log:
        if stat_vals:
            log_df = pd.DataFrame({
                "Time (s)": [f"{v:.1f}" for v in t_vals[-100:]],
                "Score": [f"{v:.3f}" for v in s_vals[-100:]],
                "EAR": [f"{v:.3f}" for v in e_vals[-100:]],
                "Status": stat_vals[-100:],
            })
            st.dataframe(
                log_df[::-1],
                use_container_width=True,
                height=280,
            )
            # Download button
            csv_str = log_df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV",
                csv_str,
                file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No data yet — start the camera.")

    # ── Row 4: System performance ────────────────────────────────────────
    st.markdown("---")
    _render_system_panel(cfg)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _render_status_panel() -> None:
    """Render the driver status + confidence + audio alarm area."""
    st.markdown(section_header_html("Driver Status", "🎯"), unsafe_allow_html=True)

    result = st.session_state.last_result or {}
    status = result.get("status", "No Face")
    confidence = result.get("confidence", 0.0)
    score = result.get("score", 0.0)
    color = result.get("color_hex", "#888888")

    # Animated status badge
    st.markdown(status_badge_html(status, confidence), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence score card
    st.markdown(
        metric_card_html("Confidence Score", f"{confidence:.0f}%",
                         f"Drowsiness score: {score:.3f}", color),
        unsafe_allow_html=True,
    )

    # Drowsiness score bar
    bar_cls = (
        "progress-danger" if score > 0.65
        else "progress-warn" if score > 0.35
        else "progress-safe"
    )
    st.markdown(
        progress_bar_html("Drowsiness Score", score, 1.0, bar_cls),
        unsafe_allow_html=True,
    )

    # Drowsy alert banner
    if status == "Drowsy":
        st.markdown("""
        <div class="alert-banner">
            ⚠️ DROWSINESS DETECTED — Please pull over safely!
        </div>
        """, unsafe_allow_html=True)

    # Alarm audio component
    alarm_active = status == "Drowsy" and st.session_state.get("alarm_enabled", True)
    st.session_state.audio_manager.render_alarm_component(
        playing=alarm_active,
        volume=st.session_state.get("alarm_volume", 0.75),
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Component scores (bar chart)
    comp_scores = result.get("component_scores", {})
    if comp_scores:
        st.markdown(
            "<div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.1em;color:#5a5a7a;margin-bottom:6px'>Score Breakdown</div>",
            unsafe_allow_html=True,
        )
        for name, val in comp_scores.items():
            pct = val * 100
            bar_html = (
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                f"<span style='font-size:0.68rem;color:#7070a0;width:70px'>{name}</span>"
                f"<div style='flex:1;background:rgba(255,255,255,0.05);border-radius:100px;"
                f"height:5px;overflow:hidden'>"
                f"<div style='height:100%;width:{pct:.0f}%;background:linear-gradient("
                f"90deg,#667eea,#764ba2);border-radius:100px'></div></div>"
                f"<span style='font-size:0.68rem;color:#9090b0;width:32px;text-align:right'>"
                f"{val:.2f}</span></div>"
            )
            st.markdown(bar_html, unsafe_allow_html=True)


def _render_eye_panel() -> None:
    """Render eye metrics section."""
    st.markdown(section_header_html("Eye Metrics", "👁"), unsafe_allow_html=True)

    result = st.session_state.last_result or {}
    ear = result.get("ear", 0.0)
    is_closed = result.get("is_closed", False)
    closure = result.get("closure_secs", 0.0)
    blink_count = result.get("blink_count", 0)
    blink_rate = result.get("blink_rate", 0.0)

    cfg = st.session_state.cfg.ear
    ear_bar_cls = "progress-danger" if ear < cfg.threshold else "progress-safe"

    st.markdown(
        progress_bar_html("EAR", ear, 0.45, ear_bar_cls),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Blinks", str(blink_count))
        eye_state = "🔴 Closed" if is_closed else "🟢 Open"
        st.metric("Eye State", eye_state)
    with c2:
        st.metric("Blink Rate", f"{blink_rate:.0f} bpm")
        st.metric("Closure", f"{closure:.1f}s")


def _render_yawn_panel() -> None:
    """Render yawn metrics section."""
    st.markdown(section_header_html("Yawn Detection", "😮"), unsafe_allow_html=True)

    result = st.session_state.last_result or {}
    mar = result.get("mar", 0.0)
    is_yawning = result.get("is_yawning", False)
    yawn_count = result.get("yawn_count", 0)

    cfg = st.session_state.cfg.mar
    mar_bar_cls = "progress-danger" if mar > cfg.threshold else "progress-safe"

    st.markdown(
        progress_bar_html("MAR", mar, 1.0, mar_bar_cls),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Yawns", str(yawn_count))
    with c2:
        state = "😮 Yawning" if is_yawning else "😐 Normal"
        st.metric("State", state)


def _render_headpose_panel() -> None:
    """Render head pose section."""
    st.markdown(section_header_html("Head Pose", "🧭"), unsafe_allow_html=True)

    result = st.session_state.last_result or {}
    pitch = result.get("pitch", 0.0)
    yaw = result.get("yaw", 0.0)
    roll = result.get("roll", 0.0)
    is_nodding = abs(pitch) > st.session_state.cfg.head_pose.pitch_threshold
    is_away = abs(yaw) > st.session_state.cfg.head_pose.yaw_threshold

    c1, c2 = st.columns(2)
    with c1:
        pitch_color = "#DC143C" if is_nodding else "#32CD32"
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.65rem;color:#7070a0'>PITCH (nod)</div>"
            f"<div style='font-size:1.5rem;font-weight:800;color:{pitch_color}'>{pitch:.1f}°</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        yaw_color = "#FFA500" if is_away else "#32CD32"
        st.markdown(
            f"<div style='text-align:center;margin-top:8px'>"
            f"<div style='font-size:0.65rem;color:#7070a0'>YAW (turn)</div>"
            f"<div style='font-size:1.5rem;font-weight:800;color:{yaw_color}'>{yaw:.1f}°</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.65rem;color:#7070a0'>ROLL (tilt)</div>"
            f"<div style='font-size:1.5rem;font-weight:800;color:#9090b0'>{roll:.1f}°</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        nod_str = "⚠️ Nodding" if is_nodding else "✓ Normal"
        away_str = "⚠️ Looking Away" if is_away else "✓ Normal"
        st.caption(nod_str)
        st.caption(away_str)


def _render_system_panel(cfg: AppConfig) -> None:
    """Render system performance metrics row."""
    st.markdown(section_header_html("System Performance", "⚡"), unsafe_allow_html=True)

    result = st.session_state.last_result or {}
    fps = result.get("fps", 0.0)
    m: MetricsTracker = st.session_state.metrics
    report = m.session_report()

    col_fps, col_frames, col_avg_fps, col_dur = st.columns(4)
    fps_color = "#32CD32" if fps >= 15 else "#FFA500" if fps >= 10 else "#DC143C"

    with col_fps:
        st.markdown(
            metric_card_html("Live FPS", f"{fps:.0f}", "target ≥ 15", fps_color),
            unsafe_allow_html=True,
        )
    with col_frames:
        st.markdown(
            metric_card_html("Frames Processed", str(report["total_frames"]), "this session"),
            unsafe_allow_html=True,
        )
    with col_avg_fps:
        st.markdown(
            metric_card_html("Avg FPS", f"{report['avg_fps']:.1f}", "session average"),
            unsafe_allow_html=True,
        )
    with col_dur:
        elapsed = int(report["session_duration_secs"])
        st.markdown(
            metric_card_html(
                "Session Duration",
                f"{elapsed//60:02d}:{elapsed%60:02d}",
                "mm:ss",
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

def _save_screenshot() -> None:
    """Placeholder for screenshot saving — captures current Streamlit state."""
    os.makedirs("data/screenshots", exist_ok=True)
    count = st.session_state.screenshot_count + 1
    st.session_state.screenshot_count = count
    st.success(f"Screenshot #{count} would be saved to data/screenshots/")


# ---------------------------------------------------------------------------
# History updater — called every autorefresh tick
# ---------------------------------------------------------------------------

def update_history_from_result() -> None:
    """Read latest result from DetectionResult slot and update history."""
    slot: DetectionResult = st.session_state.result_slot
    data = slot.get()
    if data is None:
        return

    # Deduplicate by timestamp
    prev = st.session_state.last_result
    if prev is not None and prev.get("ts") == data.get("ts"):
        return

    st.session_state.last_result = data

    # Append to lightweight lists for charts (keep last 400 points)
    start_ts = st.session_state.session_started
    rel_t = data.get("ts", time.time()) - start_ts

    MAX_HIST = 400
    for key, val in [
        ("time_history", rel_t),
        ("score_history", data.get("score", 0.0)),
        ("ear_history", data.get("ear", 0.0)),
        ("status_log", data.get("status", "No Face")),
    ]:
        lst = st.session_state[key]
        lst.append(val)
        if len(lst) > MAX_HIST:
            st.session_state[key] = lst[-MAX_HIST:]

    # Update MetricsTracker
    m: MetricsTracker = st.session_state.metrics
    m.update(
        ear=data.get("ear", 0.0),
        mar=data.get("mar", 0.0),
        pitch=data.get("pitch", 0.0),
        yaw=data.get("yaw", 0.0),
        roll=data.get("roll", 0.0),
        drowsiness_score=data.get("score", 0.0),
        status=data.get("status", "No Face"),
        confidence=data.get("confidence", 0.0),
        fps=data.get("fps", 0.0),
        face_detected=data.get("face_detected", False),
        mar_threshold=st.session_state.cfg.mar.threshold,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    # Load CSS
    inject_css()

    # Default config
    cfg: AppConfig = get_config()

    # Initialise session state
    init_session_state(cfg)

    # Render sidebar and get live settings
    settings = render_sidebar(cfg)

    # Apply settings to config
    cfg.show_landmarks = settings.get("show_landmarks", True)
    cfg.ear.threshold = settings.get("ear_threshold", cfg.ear.threshold)
    cfg.mar.threshold = settings.get("mar_threshold", cfg.mar.threshold)
    cfg.head_pose.pitch_threshold = settings.get("pitch_threshold", cfg.head_pose.pitch_threshold)
    cfg.head_pose.yaw_threshold = settings.get("yaw_threshold", cfg.head_pose.yaw_threshold)
    st.session_state.cfg = cfg
    st.session_state.alarm_enabled = settings.get("alarm_enabled", True)
    st.session_state.alarm_volume = settings.get("alarm_volume", 0.75)

    # Auto-refresh every 150 ms for near-real-time UI updates
    st_autorefresh(interval=150, limit=None, key="main_refresh")

    # Pull latest detection data into session state
    update_history_from_result()

    # Render main dashboard
    render_main(settings, cfg)


if __name__ == "__main__":
    main()
