"""
Premium dark-mode CSS for the Driver Drowsiness Detection System.

Injects a full design system into Streamlit via st.markdown:
  - Google Font: Inter
  - Deep space background
  - Glassmorphism cards
  - Gradient accent system
  - Animated status indicators (pulse ring)
  - Custom sidebar styling
  - Responsive metric cards
  - Scrollbar and input overrides
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CSS string
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   FONT IMPORT
═══════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT VARIABLES
═══════════════════════════════════════════════════════════════════════════ */
:root {
    --bg-primary:    #080812;
    --bg-secondary:  #0e0e1f;
    --bg-card:       rgba(255, 255, 255, 0.04);
    --bg-card-hover: rgba(255, 255, 255, 0.07);

    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(102, 126, 234, 0.4);

    --text-primary:   #e8e8f5;
    --text-secondary: #9090b0;
    --text-muted:     #5a5a7a;

    --accent-1: #667eea;
    --accent-2: #764ba2;
    --accent-3: #f093fb;

    --green:  #32CD32;
    --orange: #FFA500;
    --red:    #DC143C;
    --blue:   #4facfe;
    --gray:   #888888;

    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;

    --shadow-card: 0 4px 24px rgba(0,0,0,0.4), 0 1px 4px rgba(0,0,0,0.3);
    --shadow-glow: 0 0 30px rgba(102,126,234,0.2);
}

/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL RESET & BASE
═══════════════════════════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: radial-gradient(ellipse at 20% 50%, rgba(102,126,234,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(118,75,162,0.06) 0%, transparent 50%),
                linear-gradient(180deg, #080812 0%, #0a0a1a 100%) !important;
    min-height: 100vh;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(102,126,234,0.7); }

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c20 0%, #080818 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}
/* Sidebar section dividers */
section[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 0.75rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TYPOGRAPHY
═══════════════════════════════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}
p, span, label, div {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP HEADER (title area)
═══════════════════════════════════════════════════════════════════════════ */
.ddd-header {
    background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(118,75,162,0.12) 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-lg);
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: var(--shadow-glow);
}
.ddd-header-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.ddd-header-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary) !important;
    margin-top: 4px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   GLASS CARDS
═══════════════════════════════════════════════════════════════════════════ */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s, background 0.2s;
    box-shadow: var(--shadow-card);
}
.glass-card:hover {
    border-color: var(--border-accent);
    background: var(--bg-card-hover);
}
.glass-card-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-secondary) !important;
    margin-bottom: 8px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS INDICATOR (animated ring)
═══════════════════════════════════════════════════════════════════════════ */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.0rem;
    letter-spacing: 0.03em;
}
.status-alert {
    background: rgba(50,205,50,0.12);
    border: 1.5px solid rgba(50,205,50,0.5);
    color: #32CD32 !important;
}
.status-slight {
    background: rgba(255,165,0,0.12);
    border: 1.5px solid rgba(255,165,0,0.5);
    color: #FFA500 !important;
}
.status-drowsy {
    background: rgba(220,20,60,0.12);
    border: 1.5px solid rgba(220,20,60,0.5);
    color: #DC143C !important;
}
.status-noface {
    background: rgba(136,136,136,0.10);
    border: 1.5px solid rgba(136,136,136,0.3);
    color: #888888 !important;
}

/* Pulse ring animation */
@keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 0.8; }
    50%  { transform: scale(1.4); opacity: 0;   }
    100% { transform: scale(1);   opacity: 0;   }
}
.pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    position: relative;
    flex-shrink: 0;
}
.pulse-dot::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    animation: pulse-ring 1.6s ease-out infinite;
}
.pulse-green   { background: var(--green); }
.pulse-green::after   { border: 2px solid var(--green); }
.pulse-orange  { background: var(--orange); }
.pulse-orange::after  { border: 2px solid var(--orange); animation-duration: 1s; }
.pulse-red     { background: var(--red); }
.pulse-red::after     { border: 2px solid var(--red); animation-duration: 0.7s; }
.pulse-gray    { background: var(--gray); }
.pulse-gray::after    { border: 2px solid var(--gray); animation: none; }

/* ═══════════════════════════════════════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════════════════════════════════════ */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    text-align: center;
    transition: all 0.2s;
}
.metric-card:hover { border-color: var(--border-accent); }
.metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary) !important;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.metric-sub {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    margin-top: 2px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROGRESS BARS (EAR / MAR / Score)
═══════════════════════════════════════════════════════════════════════════ */
.progress-container {
    width: 100%;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0;
}
.progress-bar {
    height: 100%;
    border-radius: 100px;
    transition: width 0.2s ease;
}
.progress-safe    { background: linear-gradient(90deg, #4facfe, #00f2fe); }
.progress-warn    { background: linear-gradient(90deg, #f9d423, #ff4e50); }
.progress-danger  { background: linear-gradient(90deg, #f093fb, #f5576c); }
.progress-score   { background: linear-gradient(90deg, #667eea, #764ba2); }

/* ═══════════════════════════════════════════════════════════════════════════
   ALERT BANNER
═══════════════════════════════════════════════════════════════════════════ */
@keyframes alert-flash {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
.alert-banner {
    background: linear-gradient(135deg, rgba(220,20,60,0.2), rgba(240,20,60,0.1));
    border: 1.5px solid rgba(220,20,60,0.6);
    border-radius: var(--radius-md);
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: alert-flash 1s ease-in-out infinite;
    font-weight: 600;
    font-size: 0.95rem;
    color: #ff6b7a !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   VIDEO CONTAINER
═══════════════════════════════════════════════════════════════════════════ */
.video-container {
    border: 1.5px solid var(--border-accent);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: 0 0 40px rgba(102,126,234,0.15), var(--shadow-card);
}
/* Make WebRTC video fill its container */
.video-container video {
    width: 100% !important;
    height: auto !important;
    display: block;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STREAMLIT OVERRIDES
═══════════════════════════════════════════════════════════════════════════ */

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Sliders */
.stSlider > div[data-baseweb="slider"] > div { background: var(--border-subtle) !important; }
.stSlider [role="slider"] { background: var(--accent-1) !important; border-color: var(--accent-1) !important; }

/* Selectbox / dropdowns */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}

/* Toggle */
.stCheckbox label span[data-testid="stCheckboxLabel"] { color: var(--text-primary) !important; }

/* Expanders */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}

/* Metric (st.metric) */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
}
.stTabs [data-baseweb="tab"] { color: var(--text-secondary) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: white !important;
    border-radius: var(--radius-sm) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: var(--radius-md) !important;
}

/* Info / Warning / Error boxes */
.stAlert { border-radius: var(--radius-sm) !important; }

/* Horizontal rule */
hr { border-color: var(--border-subtle) !important; }

/* Remove Streamlit top padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Hide hamburger menu footer watermark */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""


# ---------------------------------------------------------------------------
# Reusable HTML snippets
# ---------------------------------------------------------------------------

def status_badge_html(status: str, confidence: float) -> str:
    """Return HTML for the animated status indicator."""
    css_class_map = {
        "Alert":          ("status-alert",  "pulse-green"),
        "Slightly Drowsy": ("status-slight", "pulse-orange"),
        "Drowsy":         ("status-drowsy", "pulse-red"),
        "No Face":        ("status-noface", "pulse-gray"),
    }
    card_cls, dot_cls = css_class_map.get(status, ("status-noface", "pulse-gray"))
    return f"""
    <div class="status-indicator {card_cls}">
        <span class="pulse-dot {dot_cls}"></span>
        <span>{status}</span>
        <span style="font-size:0.75rem;opacity:0.75">{confidence:.0f}%</span>
    </div>
    """


def metric_card_html(label: str, value: str, sub: str = "", color: str = "#667eea") -> str:
    """Return HTML for a single metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def progress_bar_html(label: str, value: float, max_val: float, css_class: str = "progress-safe") -> str:
    """Return HTML for a labelled progress bar."""
    pct = min(100, max(0, value / max_val * 100))
    return f"""
    <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px">
            <span style="font-size:0.72rem;color:var(--text-secondary)">{label}</span>
            <span style="font-size:0.72rem;color:var(--text-primary);font-weight:600">{value:.3f}</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar {css_class}" style="width:{pct:.1f}%"></div>
        </div>
    </div>
    """


def section_header_html(title: str, icon: str = "") -> str:
    """Return HTML for a glass-card section header."""
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        {"<span style='font-size:1.1rem'>" + icon + "</span>" if icon else ""}
        <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.12em;color:var(--text-secondary)">{title}</span>
    </div>
    """


def inject_css() -> None:
    """Inject the full CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
