"""
Session-state helpers.

Streamlit reruns the whole script top-to-bottom on every interaction, so
anything that must survive a rerun - the last result, the request history,
the backend URL the evaluator typed in - has to live in st.session_state.
Centralising that here keeps app.py free of `if "x" not in st.session_state`
boilerplate.
"""

import datetime
import streamlit as st

from config import MAX_HISTORY

_SIMPLE_DEFAULTS = {
    "result": None,
    "trip": None,
    "last_latency_ms": None,
    "history": [],
    "_health": None,
}


def init_state(default_api_url: str):
    if st.session_state.get("api_url") is None:
        st.session_state["api_url"] = default_api_url
    for key, value in _SIMPLE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = [] if value == [] else value


def log_run(trip_id: str, status: str, latency_ms, note: str = ""):
    """Prepend one request to the session's activity log (most recent first)."""
    st.session_state.setdefault("history", [])
    st.session_state["history"].insert(0, {
        "trip_id": trip_id,
        "status": status,
        "latency_ms": latency_ms,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "note": note,
    })
    st.session_state["history"] = st.session_state["history"][:MAX_HISTORY]
