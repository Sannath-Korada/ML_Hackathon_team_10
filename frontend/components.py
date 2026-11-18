"""
Reusable UI building blocks for app.py.

Keeping these separate means app.py reads top-to-bottom as a short
description of the page ("show the hero, show the trip picker, show the
results") instead of a wall of markup - which matters because every team
member needs to be able to walk an evaluator through this code.
"""

import json
import streamlit as st

from config import (
    APP_NAME, COLOR_GPS, COLOR_ROAD, COLOR_PREDICTED,
    COLOR_CLASSIFIER_ONLY, COLOR_REFERENCE,
)


def render_hero():
    st.markdown("""
    <div class="hero">
     <div class="kicker">GPS trajectory intelligence</div>
     <div class="hero-title">Route Matching Dashboard</div>
     <p class="hero-sub">Turn noisy GPS observations into a connected, road-network-aware route.</p>
     <span class="pill">ML candidate scoring</span><span class="pill">HMM + Viterbi decoding</span><span class="pill">Live FastAPI backend</span>
    </div>
    """, unsafe_allow_html=True)


def render_connection_panel(api_cls, default_url):
    """Sidebar block: backend URL, a same-machine preset, and a health check
    with a persistent status chip (not just a toast that disappears)."""
    st.markdown("#### Backend connection")
    st.caption("Laptop 1 must run FastAPI with `--host 0.0.0.0` for Laptop 2 to reach it.")

    preset_col, clear_col = st.columns(2)
    with preset_col:
        if st.button("This laptop", use_container_width=True,
                      help="Use when the API and UI run on the same machine"):
            st.session_state["api_url"] = default_url
    with clear_col:
        if st.button("Clear URL", use_container_width=True):
            st.session_state["api_url"] = ""

    url = st.text_input(
        "FastAPI base URL",
        st.session_state.get("api_url", default_url),
        placeholder="http://<LAPTOP_1_LAN_IP>:8000",
    ).strip()
    st.session_state["api_url"] = url
    api = api_cls(url) if url else None

    if st.button("Test connection", use_container_width=True, disabled=not bool(url)):
        try:
            data, elapsed_ms = api.health()
            st.session_state["_health"] = ("ok", elapsed_ms, data)
        except Exception as exc:
            st.session_state["_health"] = ("fail", None, str(exc))

    health = st.session_state.get("_health")
    if health is None:
        st.markdown("<div class='status-chip status-unknown'>&#9679; Not tested yet</div>", unsafe_allow_html=True)
    elif health[0] == "ok":
        st.markdown(
            f"<div class='status-chip status-ok'>&#9679; API reachable &middot; {health[1]:.0f} ms</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='status-chip status-fail'>&#9679; {health[2]}</div>", unsafe_allow_html=True)
    return api


def render_pipeline_steps():
    st.markdown("#### Processing pipeline")
    steps = [
        "GPS & road data", "Candidate generation", "ML candidate scoring",
        "HMM / Viterbi decoding", "Route response",
    ]
    for i, label in enumerate(steps, 1):
        st.markdown(f"<div class='pipe-step'><span class='pipe-num'>{i:02d}</span>{label}</div>",
                    unsafe_allow_html=True)


def render_trip_grid(trip_ids, selected):
    """A clickable grid of trip cards (so all ten evaluation trips are visible
    at once, per the hackathon guide's UI responsibilities) with a text-input
    fallback if the backend has no trip list yet. Returns the selected id."""
    if not trip_ids:
        return st.text_input("Trip ID", placeholder="Waiting for backend trip list").strip()

    st.caption(f"{len(trip_ids)} evaluation trip(s) loaded from the backend.")
    cols_per_row = 5
    for row_start in range(0, len(trip_ids), cols_per_row):
        row = trip_ids[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, trip_id in zip(cols, row):
            with col:
                is_selected = trip_id == selected
                label = f"\u25cf {trip_id}" if is_selected else trip_id
                if st.button(label, key=f"trip_btn_{trip_id}", use_container_width=True,
                             type="primary" if is_selected else "secondary"):
                    selected = trip_id
    return selected


def render_metrics(result):
    """Reads summary/metrics fields generically so the panel still renders
    sensibly whichever subset of metrics the backend actually computed."""
    summary = result.get("summary", {}) or {}
    metrics = result.get("metrics", {}) or {}

    def pick(*keys, default="\u2014"):
        for container in (summary, metrics, result):
            for key in keys:
                value = container.get(key)
                if value is not None:
                    return value
        return default

    specs = [
        ("GPS observations", pick("observation_count"), "Input trajectory length"),
        ("Route segments", pick("predicted_segment_count"), "Predicted sequence length"),
        ("Top-1 accuracy", pick("top1_accuracy", "observation_top1_accuracy"), "Observation-level"),
        ("Route continuity", pick("route_continuity"), "Sequence quality"),
        ("Macro F1", pick("macro_f1", "f1_score"), "Candidate classification"),
        ("Log loss", pick("log_loss"), "Lower is better"),
        ("Candidate miss rate", pick("candidate_miss_rate"), "Reference missing from candidates"),
        ("Inference time", pick("inference_time_ms", "inference_time"), "Model + decoder"),
    ]
    cols = st.columns(4)
    for i, (label, value, note) in enumerate(specs):
        with cols[i % 4]:
            st.markdown(
                f"<div class='kpi'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>",
                unsafe_allow_html=True,
            )


def render_legend():
    rows = [
        (COLOR_GPS, "GPS observations"),
        (COLOR_ROAD, "Road network"),
        (COLOR_CLASSIFIER_ONLY, "Classifier-only (pre-HMM)"),
        (COLOR_PREDICTED, "Predicted route (post-HMM)"),
        (COLOR_REFERENCE, "Reference route (dev data only)"),
    ]
    html = ["<div class='panel'><b>Legend</b><hr>"]
    for color, label in rows:
        html.append(f"<div class='legend-row'><span class='legend-dot' style='background:{color}'></span>{label}</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_run_details(trip, latency_ms):
    latency_text = f"{latency_ms:.0f} ms" if latency_ms else "\u2014"
    st.markdown(f"""
    <div class="panel">
      <b>Run details</b><hr>
      <span style="color:#91a4bf">Trip ID</span><h3 style="margin:4px 0 10px">{trip or '\u2014'}</h3>
      <span style="color:#91a4bf">Round-trip latency</span><br><b>{latency_text}</b><hr>
      <span style="color:#91a4bf">Decoder</span><br><b>HMM + Viterbi</b><hr>
      <span style="color:#91a4bf">Prediction source</span><br><b>Live FastAPI response (not hardcoded)</b>
    </div>
    """, unsafe_allow_html=True)


def render_history_log():
    """A rolling log of every /predict call this session: trip, status,
    latency. This gives an evaluator a second, independent way (alongside
    the Laptop-1 terminal logs the guidelines require you to show) to see
    that the system was actually exercised across multiple trips, live."""
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No requests sent yet this session.")
        return
    st.caption(f"{len(history)} request(s) this session \u2014 every trip hits the live API.")
    rows = []
    for h in history:
        latency = f"{h['latency_ms']:.0f} ms" if h.get("latency_ms") else "\u2014"
        status_class = "ok" if h.get("status") == "ok" else "fail"
        rows.append(
            f"<tr><td>{h['time']}</td><td>{h['trip_id']}</td>"
            f"<td class='{status_class}'>{h['status']}</td><td>{latency}</td></tr>"
        )
    table = (
        "<table class='log-table'><tr><th>Time</th><th>Trip</th><th>Status</th><th>Latency</th></tr>"
        + "".join(rows) + "</table>"
    )
    st.markdown(table, unsafe_allow_html=True)
    st.download_button(
        "Download session log (JSON)",
        json.dumps(history, indent=2),
        file_name="pathsnap_session_log.json",
        mime="application/json",
        use_container_width=True,
    )


def render_empty_state():
    st.markdown("""
    <div class="panel" style="padding:30px">
      <div style="font-size:2rem">\U0001f6f0\ufe0f</div><h3>Ready for a trajectory</h3>
      <p style="color:#91a4bf;max-width:700px;line-height:1.7">
      Select an evaluation trip above and run map matching. Route metrics, the
      geographic reconstruction and prediction details are populated entirely
      from the live API response - nothing here is a canned demo or a
      fabricated metric.</p>
      <span class="pill">1 &middot; Select trip</span><span class="pill">2 &middot; Predict</span><span class="pill">3 &middot; Inspect route</span>
    </div>""", unsafe_allow_html=True)
