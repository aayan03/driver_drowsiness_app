"""
Live webcam capture Streamlit component.
Uses the browser getUserMedia API to stream frames to Python
without requiring a manual click.
"""
from __future__ import annotations
import os
import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(__file__), "live_camera")
_func = components.declare_component("live_camera", path=_DIR)


def live_camera_input(height: int = 420, key: str = "live_camera"):
    """
    Renders a live webcam component.
    Returns the latest frame as a base64 JPEG data-URL string,
    or None if camera is not yet ready.
    """
    return _func(height=height, key=key, default=None)
