"""
AI Driver Drowsiness Detection System
Tornado shim at top fixes BaseAsyncIOLoop removal in tornado >=6.4.
Two capture modes: WebRTC (local) and Snapshot/st.camera_input (cloud).
"""
from __future__ import annotations

# ── Tornado >=6.4 compatibility shim ─────────────────────────────────────
import tornado.platform.asyncio as _tpa
if not hasattr(_tpa, "BaseAsyncIOLoop"):
    import asyncio as _asyncio
    class _BaseAsyncIOLoopShim:
        def __init__(self, loop=None):
            self.asyncio_loop = loop or _asyncio.get_event_loop()
    _tpa.BaseAsyncIOLoop = _BaseAsyncIOLoopShim
# ─────────────────────────────────────────────────────────────────────────

import json
import logging
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
from models.drowsiness_model import (
    DrowsinessClassifier,
    DrowsinessResult,
    draw_status_overlay,
)
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Drowsiness Detection",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_face_mesh(det_conf: float, track_conf: float):
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )


class _ResultSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Optional[dict] = None

    def put(self, data: dict):
        with self._lock:
            self._data = data

    def get(self) -> Optional[dict]:
        with self._lock:
            return self._data


class FrameProcessor:
    """Full detection pipeline for a single BGR frame."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.eye = EyeDetector(cfg.ear.threshold, cfg.ear.smoothing_alpha,
                                cfg.ear.consec_frames)
        self.yawn = YawnDetector(cfg.mar.threshold, cfg.mar.smoothing_alpha,
                                  cfg.mar.consec_frames)
        self.pose = HeadPoseDetector(cfg.head_pose.pitch_threshold,
                                     cfg.head_pose.yaw_threshold,
                                     cfg.head_pose.roll_threshold)
        self.clf = DrowsinessClassifier(cfg.ear, cfg.mar,
                                         cfg.head_pose, cfg.drowsiness)
        self.face_mesh = load_face_mesh(cfg.face_min_confidence,
                                         cfg.face_tracking_confidence)
        self._prev = time.time()
        self.fps = 0.0

    def process(self, bgr: np.ndarray) -> tuple:
        now = time.time()
        dt = now - self._prev
        self.fps = 1.0 / dt if dt > 0 else 30.0
        self._prev = now

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            eye_r = self.eye.process(lm, (h, w))
            yawn_r = self.yawn.process(lm, (h, w))
            pose_r = self.pose.process(lm, (h, w))
            dr: DrowsinessResult = self.clf.evaluate(eye_r, yawn_r, pose_r)

            annotated = draw_status_overlay(
                bgr.copy(), dr, self.fps,
                show_landmarks=self.cfg.show_landmarks,
                eye_coords=(eye_r["left_coords"], eye_r["right_coords"]),
                mouth_coords=yawn_r["mouth_coords"],
                nose_2d=pose_r.get("nose_2d"),
                nose_end=pose_r.get("nose_end"),
            )
            data = dict(
                face_detected=True, status=dr.status, score=dr.score,
                confidence=dr.confidence, ear=dr.ear, mar=dr.mar,
                pitch=dr.pitch, yaw=dr.yaw, roll=dr.roll,
                is_closed=dr.is_closed, closure_secs=dr.closure_secs,
                is_yawning=dr.is_yawning, blink_rate=dr.blink_rate,
                blink_count=eye_r["blink_count"],
                yawn_count=yawn_r["yawn_count"],
                color_hex=dr.color_hex, fps=round(self.fps, 1),
                component_scores=dr.component_scores, ts=now,
            )
        else:
            annotated = bgr.copy()
            cv2.putText(annotated, "No Face Detected",
                        (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                        0.9, (100, 100, 120), 2, cv2.LINE_AA)
            data = dict(
                face_detected=False, status="No Face", score=0.0,
                confidence=0.0, ear=0.0, mar=0.0, pitch=0.0, yaw=0.0,
                roll=0.0, is_closed=False, closure_secs=0.0,
                is_yawning=False, blink_rate=0.0, blink_count=0,
                yawn_count=0, color_hex="#888888",
                fps=round(self.fps, 1), component_scores={}, ts=now,
            )
        return annotated, data


class DrowsinessVideoProcessor:
    def __init__(self, cfg: AppConfig, slot: _ResultSlot):
        self.proc = FrameProcessor(cfg)
        self.slot = slot

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        annotated, data = self.proc.process(bgr)
        self.slot.put(data)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def init_state(cfg: AppConfig):
    defaults = {
        "cfg": cfg,
        "result_slot": _ResultSlot(),
        "processor": None,
        "metrics": MetricsTracker(cfg.history_size, cfg.log_csv_path,
                                   cfg.ear.threshold),
        "audio": AudioManager(),
        "alarm_enabled": True,
        "alarm_volume": 0.75,
        "last_result": None,
        "score_history": [], "ear_history": [],
        "time_history": [], "status_log": [],
        "session_start": time.time(),
        "snap_frame_count": 0,
        "capture_mode": "snapshot",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _chart_base():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#9090b0"),
        margin=dict(l=40, r=20, t=20, b=36),
        showlegend=False,
    )


def score_chart(t, s):
    fig = go.Figure()
    for y, lbl, col in [(0.35, "Alert", "rgba(50,205,50,.3)"),
                         (0.65, "Drowsy", "rgba(220,20,60,.3)")]:
        fig.add_hline(y=y, line_dash="dot", line_color=col, opacity=.7,
                      annotation_text=lbl,
                      annotation_font_color=col, annotation_font_size=10)
    fig.add_trace(go.Scatter(x=t, y=s, mode="lines", name="Score",
                              line=dict(color="#667eea", width=2),
                              fill="tozeroy",
                              fillcolor="rgba(102,126,234,.08)"))
    fig.update_layout(height=200, **_chart_base(),
                      yaxis=dict(range=[0, 1], showgrid=True,
                                  gridcolor="rgba(255,255,255,.05)"),
                      xaxis=dict(showgrid=True,
                                  gridcolor="rgba(255,255,255,.05)"))
    return fig


def ear_chart(t, e, thr):
    fig = go.Figure()
    fig.add_hline(y=thr, line_dash="dot", line_color="rgba(255,165,0,.6)",
                   annotation_text="Closed threshold",
                   annotation_font_color="rgba(255,165,0,.8)",
                   annotation_font_size=10)
    fig.add_trace(go.Scatter(x=t, y=e, mode="lines",
                              line=dict(color="#4facfe", width=2),
                              fill="tozeroy",
                              fillcolor="rgba(79,172,254,.08)"))
    fig.update_layout(height=180, **_chart_base(),
                      yaxis=dict(range=[0, .5], showgrid=True,
                                  gridcolor="rgba(255,255,255,.05)"),
                      xaxis=dict(showgrid=True,
                                  gridcolor="rgba(255,255,255,.05)"))
    return fig


def donut_chart(a, sl, d):
    fig = go.Figure(go.Pie(
        labels=["Alert", "Slightly Drowsy", "Drowsy"],
        values=[max(.01, a), max(.01, sl), max(.01, d)],
        hole=.65,
        marker=dict(colors=["#32CD32", "#FFA500", "#DC143C"],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=170, plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#9090b0"),
        legend=dict(font=dict(size=10, color="#9090b0"),
                    bgcolor="rgba(0,0,0,0)", x=1.0, y=0.5),
        margin=dict(l=0, r=80, t=10, b=10),
        annotations=[dict(text=str(round(a)) + "%",
                          x=.5, y=.5, font_size=14, showarrow=False,
                          font=dict(color="#32CD32"))],
    )
    return fig


def render_sidebar(cfg: AppConfig) -> dict:
    s = {}
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:8px 0 16px">
  <div style="font-size:2rem">&#x1F697;</div>
  <div style="font-size:1rem;font-weight:800;
       background:linear-gradient(135deg,#667eea,#f093fb);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
       background-clip:text">DrowseSafe AI</div>
  <div style="font-size:.65rem;color:#5a5a7a">Driver Safety System</div>
</div><hr>
""", unsafe_allow_html=True)

        st.markdown("**Capture Mode**")
        mode = st.radio(
            "Mode",
            ["Snapshot (Cloud)", "WebRTC (Local)"],
            index=0, label_visibility="collapsed",
            help="Snapshot works everywhere. WebRTC needs local network.",
        )
        s["capture_mode"] = "snapshot" if "Snapshot" in mode else "webrtc"
        s["show_landmarks"] = st.toggle("Show landmarks", value=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        with st.expander("Detection Thresholds"):
            s["ear_threshold"] = st.slider("EAR closed", .15, .40,
                                            cfg.ear.threshold, .01)
            s["mar_threshold"] = st.slider("MAR yawn", .40, .80,
                                            cfg.mar.threshold, .01)
            s["pitch_threshold"] = st.slider("Pitch deg", 5., 40.,
                                              cfg.head_pose.pitch_threshold, 1.)
            s["yaw_threshold"] = st.slider("Yaw deg", 10., 50.,
                                            cfg.head_pose.yaw_threshold, 1.)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Alarm**")
        s["alarm_enabled"] = st.toggle("Enable alarm", value=True)
        s["alarm_volume"] = st.slider("Volume", 0., 1., .75, .05)

        uploaded = st.file_uploader("Custom alarm (mp3/wav/ogg)",
                                     type=["mp3", "wav", "ogg"])
        if uploaded:
            mime = ("audio/mpeg" if uploaded.name.endswith(".mp3")
                    else "audio/ogg" if uploaded.name.endswith(".ogg")
                    else "audio/wav")
            st.session_state.audio.load_from_bytes(uploaded.read(), mime)
            st.success("Custom alarm loaded")

        st.session_state.audio.render_test_button_component(s["alarm_volume"])
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("**Session**")
        m: MetricsTracker = st.session_state.metrics
        rep = m.session_report()
        c1, c2 = st.columns(2)
        with c1:
            e = int(rep["session_duration_secs"])
            st.metric("Duration", f"{e//60:02d}:{e%60:02d}")
        with c2:
            st.metric("Blinks", str(rep["total_blinks"]))

        if st.button("Export CSV", use_container_width=True):
            st.success(f"Saved: {m.export_csv()}")

        rjson = json.dumps(rep, indent=2)
        st.download_button(
            "Download Report", rjson,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        if st.button("Reset Session", use_container_width=True):
            for k in ("metrics", "score_history", "ear_history",
                       "time_history", "status_log", "last_result"):
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("DrowseSafe AI v1.0 | MediaPipe Face Mesh | EAR+MAR+Pose")
    return s


def render_snapshot_mode(cfg: AppConfig):
    st.markdown(section_header_html("Live Camera Feed", "Camera"),
                unsafe_allow_html=True)

    if st.session_state.processor is None:
        st.session_state.processor = FrameProcessor(cfg)
    proc: FrameProcessor = st.session_state.processor

    proc.eye.ear_threshold = cfg.ear.threshold
    proc.yawn.mar_threshold = cfg.mar.threshold
    proc.pose.pitch_threshold = cfg.head_pose.pitch_threshold
    proc.pose.yaw_threshold = cfg.head_pose.yaw_threshold
    proc.clf.ear_cfg = cfg.ear
    proc.clf.mar_cfg = cfg.mar
    proc.clf.pose_cfg = cfg.head_pose
    proc.clf.cfg = cfg.drowsiness
    proc.cfg = cfg

    count = st.session_state.snap_frame_count
    img_file = st.camera_input(
        "Camera",
        key=f"snap_{count % 5}",
        label_visibility="collapsed",
    )

    if img_file is not None:
        pil = Image.open(img_file).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        annotated, data = proc.process(bgr)
        st.session_state.result_slot.put(data)
        rgb_out = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_out, channels="RGB", use_column_width=True)
        st.session_state.snap_frame_count += 1
    else:
        st.info("Allow camera access — your annotated feed will appear here.")


def render_webrtc_mode(cfg: AppConfig):
    st.markdown(section_header_html("Live Camera Feed", "Camera"),
                unsafe_allow_html=True)
    slot: _ResultSlot = st.session_state.result_slot

    ctx = webrtc_streamer(
        key="ddd_webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        video_processor_factory=lambda: DrowsinessVideoProcessor(cfg, slot),
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                      "frameRate": {"ideal": 30}},
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.state.playing:
        st.markdown(
            "<div style='font-size:.72rem;color:#32CD32;margin-top:6px'>"
            "Camera active</div>", unsafe_allow_html=True)
    else:
        st.info("Click START to begin monitoring.")
        st.caption("If camera does not connect, switch to Snapshot mode in the sidebar.")


def render_status():
    st.markdown(section_header_html("Driver Status", "Status"),
                unsafe_allow_html=True)
    r = st.session_state.last_result or {}
    status = r.get("status", "No Face")
    conf   = r.get("confidence", 0.0)
    score  = r.get("score", 0.0)
    color  = r.get("color_hex", "#888888")

    st.markdown(status_badge_html(status, conf), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(metric_card_html("Confidence", f"{conf:.0f}%",
                                  f"Score: {score:.3f}", color),
                unsafe_allow_html=True)

    bar_cls = ("progress-danger" if score > .65
                else "progress-warn" if score > .35 else "progress-safe")
    st.markdown(progress_bar_html("Drowsiness Score", score, 1.0, bar_cls),
                unsafe_allow_html=True)

    if status == "Drowsy":
        st.markdown(
            "<div class='alert-banner'>DROWSINESS DETECTED - Pull over safely!</div>",
            unsafe_allow_html=True)

    alarm_on = status == "Drowsy" and st.session_state.alarm_enabled
    st.session_state.audio.render_alarm_component(
        alarm_on, st.session_state.alarm_volume)

    cs = r.get("component_scores", {})
    if cs:
        st.markdown(
            "<div style='font-size:.65rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.1em;color:#5a5a7a;margin:12px 0 6px'>Score Breakdown</div>",
            unsafe_allow_html=True)
        for name, val in cs.items():
            pct = val * 100
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                f"<span style='font-size:.68rem;color:#7070a0;width:74px'>{name}</span>"
                f"<div style='flex:1;background:rgba(255,255,255,.05);border-radius:100px;"
                f"height:5px;overflow:hidden'><div style='height:100%;width:{pct:.0f}%;"
                f"background:linear-gradient(90deg,#667eea,#764ba2);border-radius:100px'>"
                f"</div></div><span style='font-size:.68rem;color:#9090b0;width:32px;"
                f"text-align:right'>{val:.2f}</span></div>",
                unsafe_allow_html=True)


def render_eye():
    st.markdown(section_header_html("Eye Metrics", "Eye"), unsafe_allow_html=True)
    r = st.session_state.last_result or {}
    ear = r.get("ear", 0.)
    thr = st.session_state.cfg.ear.threshold
    st.markdown(
        progress_bar_html("EAR", ear, .45,
                           "progress-danger" if ear < thr else "progress-safe"),
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Blinks", str(r.get("blink_count", 0)))
        st.metric("Eye", "Closed" if r.get("is_closed") else "Open")
    with c2:
        st.metric("Blink Rate", f"{r.get('blink_rate', 0.):.0f} bpm")
        st.metric("Closure", f"{r.get('closure_secs', 0.):.1f}s")


def render_yawn():
    st.markdown(section_header_html("Yawn Detection", "Yawn"), unsafe_allow_html=True)
    r = st.session_state.last_result or {}
    mar = r.get("mar", 0.)
    thr = st.session_state.cfg.mar.threshold
    st.markdown(
        progress_bar_html("MAR", mar, 1.0,
                           "progress-danger" if mar > thr else "progress-safe"),
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Yawns", str(r.get("yawn_count", 0)))
    with c2:
        st.metric("State", "Yawning" if r.get("is_yawning") else "Normal")


def render_pose():
    st.markdown(section_header_html("Head Pose", "Pose"), unsafe_allow_html=True)
    r = st.session_state.last_result or {}
    p = r.get("pitch", 0.)
    y = r.get("yaw", 0.)
    ro = r.get("roll", 0.)
    pthr = st.session_state.cfg.head_pose.pitch_threshold
    ythr = st.session_state.cfg.head_pose.yaw_threshold
    pc = "#DC143C" if abs(p) > pthr else "#32CD32"
    yc = "#FFA500" if abs(y) > ythr else "#32CD32"
    c1, c2 = st.columns(2)
    with c1:
        for lbl, val, col in [("PITCH", p, pc), ("YAW", y, yc)]:
            st.markdown(
                f"<div style='text-align:center;margin-bottom:6px'>"
                f"<div style='font-size:.65rem;color:#7070a0'>{lbl}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{col}'>"
                f"{val:.1f} deg</div></div>",
                unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:.65rem;color:#7070a0'>ROLL</div>"
            f"<div style='font-size:1.4rem;font-weight:800;color:#9090b0'>"
            f"{ro:.1f} deg</div></div>",
            unsafe_allow_html=True)
        st.caption("Nodding" if abs(p) > pthr else "Normal")
        st.caption("Looking Away" if abs(y) > ythr else "Normal")


def render_system():
    st.markdown(section_header_html("System Performance", "Perf"),
                unsafe_allow_html=True)
    r = st.session_state.last_result or {}
    fps = r.get("fps", 0.)
    rep = st.session_state.metrics.session_report()
    fc = "#32CD32" if fps >= 15 else "#FFA500" if fps >= 8 else "#DC143C"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card_html("Live FPS", f"{fps:.0f}", "target 15", fc),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("Frames", str(rep["total_frames"])),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("Avg FPS", f"{rep['avg_fps']:.1f}"),
                    unsafe_allow_html=True)
    with c4:
        e = int(rep["session_duration_secs"])
        st.markdown(metric_card_html("Duration", f"{e//60:02d}:{e%60:02d}"),
                    unsafe_allow_html=True)


def update_history():
    slot: _ResultSlot = st.session_state.result_slot
    data = slot.get()
    if data is None:
        return
    prev = st.session_state.last_result
    if prev and prev.get("ts") == data.get("ts"):
        return
    st.session_state.last_result = data

    start = st.session_state.session_start
    rel_t = data.get("ts", time.time()) - start
    MAX = 400
    for key, val in [("time_history", rel_t),
                      ("score_history", data.get("score", 0.)),
                      ("ear_history", data.get("ear", 0.)),
                      ("status_log", data.get("status", "No Face"))]:
        lst = st.session_state[key]
        lst.append(val)
        if len(lst) > MAX:
            st.session_state[key] = lst[-MAX:]

    m: MetricsTracker = st.session_state.metrics
    m.update(
        ear=data.get("ear", 0.), mar=data.get("mar", 0.),
        pitch=data.get("pitch", 0.), yaw=data.get("yaw", 0.),
        roll=data.get("roll", 0.), drowsiness_score=data.get("score", 0.),
        status=data.get("status", "No Face"),
        confidence=data.get("confidence", 0.),
        fps=data.get("fps", 0.), face_detected=data.get("face_detected", False),
        mar_threshold=st.session_state.cfg.mar.threshold,
    )


def main():
    inject_css()
    cfg = get_config()
    init_state(cfg)

    settings = render_sidebar(cfg)
    cfg.show_landmarks = settings.get("show_landmarks", True)
    cfg.ear.threshold = settings.get("ear_threshold", cfg.ear.threshold)
    cfg.mar.threshold = settings.get("mar_threshold", cfg.mar.threshold)
    cfg.head_pose.pitch_threshold = settings.get("pitch_threshold",
                                                  cfg.head_pose.pitch_threshold)
    cfg.head_pose.yaw_threshold = settings.get("yaw_threshold",
                                                cfg.head_pose.yaw_threshold)
    st.session_state.cfg = cfg
    st.session_state.alarm_enabled = settings.get("alarm_enabled", True)
    st.session_state.alarm_volume = settings.get("alarm_volume", .75)
    mode = settings.get("capture_mode", "snapshot")
    st.session_state.capture_mode = mode

    st_autorefresh(interval=300, limit=None, key="main_refresh")
    update_history()

    st.markdown("""
<div class="ddd-header">
  <span style="font-size:2.2rem">&#x1F697;</span>
  <div>
    <div class="ddd-header-title">AI Driver Drowsiness Detection</div>
    <div class="ddd-header-subtitle">
      Real-time Eye &middot; Yawn &middot; Head-Pose Analysis
      &nbsp;&middot;&nbsp; MediaPipe Face Mesh &nbsp;&middot;&nbsp; Production Ready
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_vid, col_status = st.columns([3, 2], gap="large")
    with col_vid:
        if mode == "snapshot":
            render_snapshot_mode(cfg)
        else:
            render_webrtc_mode(cfg)
    with col_status:
        render_status()

    st.markdown("---")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        render_eye()
    with c2:
        render_yawn()
    with c3:
        render_pose()

    st.markdown("---")
    t_vals = st.session_state.time_history
    s_vals = st.session_state.score_history
    e_vals = st.session_state.ear_history
    st_vals = st.session_state.status_log

    tab_sc, tab_ear, tab_dist, tab_log = st.tabs([
        "Drowsiness Score", "EAR Trend", "State Distribution", "Event Log",
    ])
    with tab_sc:
        if len(s_vals) > 5:
            st.plotly_chart(score_chart(t_vals[-300:], s_vals[-300:]),
                            use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.caption("Start the camera to populate the chart.")
    with tab_ear:
        if len(e_vals) > 5:
            st.plotly_chart(ear_chart(t_vals[-300:], e_vals[-300:],
                                       cfg.ear.threshold),
                            use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.caption("Start the camera to populate the chart.")
    with tab_dist:
        rep = st.session_state.metrics.session_report()
        d1, d2 = st.columns([1, 1])
        with d1:
            st.plotly_chart(donut_chart(rep["alert_pct"],
                                         rep["slight_pct"],
                                         rep["drowsy_pct"]),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with d2:
            st.markdown("<br>", unsafe_allow_html=True)
            for lbl, key, col in [
                ("Alert", "alert_pct", "#32CD32"),
                ("Slightly Drowsy", "slight_pct", "#FFA500"),
                ("Drowsy", "drowsy_pct", "#DC143C"),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"margin:4px 0'><span style='font-size:.8rem;color:{col}'>"
                    f"{lbl}</span><span style='font-size:.8rem;font-weight:700;"
                    f"color:{col}'>{rep[key]:.1f}%</span></div>",
                    unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Blink Rate", f"{rep['blink_rate_bpm']} bpm")
            st.metric("Yawns", str(rep["total_yawns"]))
            st.metric("Peak Score", f"{rep['peak_score']:.3f}")
    with tab_log:
        if st_vals:
            df = pd.DataFrame({
                "Time (s)": [f"{v:.1f}" for v in t_vals[-100:]],
                "Score": [f"{v:.3f}" for v in s_vals[-100:]],
                "EAR": [f"{v:.3f}" for v in e_vals[-100:]],
                "Status": st_vals[-100:],
            })
            st.dataframe(df[::-1], use_container_width=True, height=260)
            st.download_button(
                "Download CSV", df.to_csv(index=False),
                file_name=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No data yet.")

    st.markdown("---")
    render_system()


if __name__ == "__main__":
    main()
