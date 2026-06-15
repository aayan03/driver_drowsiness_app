# 🚗 AI Driver Drowsiness Detection System

> **Real-time driver safety monitoring powered by MediaPipe Face Mesh, Streamlit WebRTC, and a multi-feature drowsiness scoring engine.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red.svg)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10%2B-green.svg)](https://mediapipe.dev)

---

## Features

- **Real-time webcam detection** via WebRTC (works in browser — no extra client software)
- **Eye Aspect Ratio (EAR)** — detects eye closure and micro-sleeps
- **Blink frequency** — flags abnormal blink rates as fatigue signals
- **Mouth Aspect Ratio (MAR)** — detects yawning
- **Head pose estimation** — detects nodding and looking away (solvePnP)
- **Multi-feature weighted scoring** → Alert / Slightly Drowsy / Drowsy + confidence %
- **Hysteresis state machine** — prevents rapid state toggling
- **Audio alarm** — browser-native JS alarm, loops until driver recovers
- **Custom alarm upload** — MP3, WAV, or OGG
- **Premium dark UI** — glassmorphism, animated indicators, real-time charts
- **CSV export & session reports**
- **Multi-camera support** (switch via sidebar)
- **Deployable on Streamlit Cloud** (zero GPU, MediaPipe CPU inference)

---

## Project Structure

```
driver_drowsiness_app/
│
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← Python dependencies
├── packages.txt                ← System packages (Streamlit Cloud)
├── README.md
│
├── assets/
│   └── default_alarm.wav       ← Auto-generated on first run
│
├── models/
│   └── drowsiness_model.py     ← Weighted classifier + overlay drawing
│
├── detectors/
│   ├── eye_detector.py         ← EAR + blink tracking
│   ├── yawn_detector.py        ← MAR + yawn counting
│   └── headpose_detector.py    ← solvePnP Euler angles
│
├── utils/
│   ├── config.py               ← All thresholds & constants
│   ├── metrics.py              ← History, stats, CSV logging
│   └── audio_manager.py        ← Browser audio component
│
├── styles/
│   └── custom_css.py           ← Dark theme + glassmorphism CSS
│
└── data/                       ← Session CSV logs & screenshots
```

---

## Local Setup Guide

### 1. Install Python 3.9 or newer

Download from [python.org](https://www.python.org/downloads/).
Verify: `python --version`

### 2. Clone or download this project

```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection
```

### 3. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> On macOS with Apple Silicon: `pip install mediapipe` may require Rosetta. Use
> `arch -x86_64 pip install mediapipe` if you encounter issues.

### 5. Run the application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

Click **START** in the video panel and allow camera access in your browser.

---

## GitHub Setup Guide

### 1. Create a new repository

Go to [github.com/new](https://github.com/new), name it `drowsiness-detection`, and leave it empty (no README).

### 2. Initialize and push

```bash
cd drowsiness-detection      # your project folder
git init
git add .
git commit -m "Initial commit: AI Drowsiness Detection System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/drowsiness-detection.git
git push -u origin main
```

---

## Streamlit Cloud Deployment Guide

### 1. Connect GitHub

- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub and authorize Streamlit

### 2. Create a new app

- Click **New app**
- Select your repository and branch (`main`)
- Set **Main file path** to `app.py`

### 3. Deploy

Click **Deploy!** — Streamlit Cloud will:
1. Install system packages from `packages.txt`
2. Install Python packages from `requirements.txt`
3. Launch the app

### 4. Allow camera in browser

After deployment, open the app URL. Click the camera permission prompt in your browser address bar and allow access.

### Fixing common deployment errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: mediapipe` | Ensure `mediapipe>=0.10.9` is in requirements.txt |
| `libGL.so.1: cannot open shared object` | Ensure `libgl1-mesa-glx` is in packages.txt |
| `av` build fails | Pin `av>=10.0.0` — binary wheels are available |
| Camera not starting | Check browser camera permissions; try HTTPS URL |
| `aiortc` error | Pin `aiortc>=1.6.0`; it requires `av` |
| Slow FPS on Cloud | Normal — free tier is CPU only; target ≥ 10 FPS |

---

## Detection Algorithm

```
Score = 0.40 × EAR_score
      + 0.25 × closure_duration_score
      + 0.15 × yawn_score
      + 0.12 × head_pose_score
      + 0.08 × blink_rate_score

State:
  score < 0.35  →  Alert
  0.35–0.65     →  Slightly Drowsy
  score ≥ 0.65  →  Drowsy
```

Hysteresis prevents rapid toggling: a worse state requires 1s confirmation;
recovery requires 2s confirmation.

---

## Troubleshooting

**Webcam not showing:**
- Ensure no other app is using the camera
- Try `Camera 1` or `Camera 2` in the sidebar
- Refresh the page and re-click START

**FPS too low:**
- Close other browser tabs
- Reduce browser window size (less rendering overhead)
- Lower `face_min_confidence` in config.py

**Alarm not playing:**
- Some browsers block autoplay until user interaction — click START first
- Check the volume slider in the sidebar
- Use the **Test Alarm** button to verify audio works

**False drowsy alerts:**
- Increase EAR threshold slightly (e.g. 0.28)
- Ensure good lighting (face well-lit from front)
- Adjust pitch/yaw thresholds if you wear glasses

---

## License

MIT — free to use, modify, and distribute with attribution.
