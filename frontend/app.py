import json
import streamlit as st

from config import APP_NAME, DEFAULT_API_URL, THEME_CSS
from api_client import PathSnapAPI, APIError
from map_view import render_route_map
from state import init_state, log_run
import components as ui

st.set_page_config(page_title=f"{APP_NAME} | Route Intelligence", page_icon="\U0001f9ed", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)

init_state(DEFAULT_API_URL)

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown(f"## \U0001f9ed {APP_NAME}")
    st.caption("EVALUATOR CONSOLE")
    st.divider()
    api = ui.render_connection_panel(PathSnapAPI, DEFAULT_API_URL)
    st.divider()
    ui.render_pipeline_steps()
    st.divider()
    st.markdown("#### Session activity log")
    ui.render_history_log()
    st.divider()
    st.caption("Distributed map-matching \u00b7 Two-laptop deployment")

# -------------------------------------------------------------------- hero --
ui.render_hero()

# ---------------------------------------------------------- 01 trip picker --
st.markdown('<div class="section">01 / Evaluation trip</div>', unsafe_allow_html=True)

trip_ids, trip_error = [], None
if api:
    try:
        trip_ids = api.get_trip_ids()
    except APIError as exc:
        trip_error = str(exc)

selected = ui.render_trip_grid(trip_ids, st.session_state.get("trip"))
st.session_state["trip"] = selected

run_col, status_col = st.columns([1, 3])
with run_col:
    run = st.button(
        "Run map matching \u2192", type="primary", use_container_width=True,
        disabled=not bool(selected) or not api,
    )
with status_col:
    if trip_error:
        st.info("Backend not reachable yet. The interface is ready \u2014 trip options and "
                "predictions load as soon as FastAPI is connected.")
    elif not trip_ids:
        st.info("Enter the backend URL in the sidebar and test the connection to load evaluation trips.")

if run and selected and api:
    try:
        with st.spinner(f"Matching trajectory {selected}..."):
            result, elapsed_ms = api.predict(selected)
        st.session_state["result"] = result
        st.session_state["trip"] = selected
        st.session_state["last_latency_ms"] = elapsed_ms
        log_run(selected, "ok", elapsed_ms)
        st.success(f"Prediction received from FastAPI in {elapsed_ms:.0f} ms.")
    except APIError as exc:
        log_run(selected, "fail", None, note=str(exc))
        st.error(f"Prediction failed: {exc}")

# ------------------------------------------------------------- 02/03 result --
result = st.session_state.get("result")
if result:
    st.markdown('<div class="section">02 / Prediction overview</div>', unsafe_allow_html=True)
    ui.render_metrics(result)

    st.markdown('<div class="section">03 / Geographic reconstruction</div>', unsafe_allow_html=True)
    map_col, info_col = st.columns([2.1, 1], gap="large")
    with map_col:
        fmap = render_route_map(result.get("map", {}) or {})
        if fmap:
            from streamlit_folium import st_folium
            st_folium(fmap, use_container_width=True, height=560, returned_objects=[])
        else:
            st.warning(
                "No map coordinates returned for this trip yet. Map fields are read directly "
                "from the `map` object in the API response \u2014 see API_CONTRACT.md."
            )
    with info_col:
        ui.render_run_details(st.session_state.get("trip"), st.session_state.get("last_latency_ms"))
        ui.render_legend()
        st.download_button(
            "Download response JSON", json.dumps(result, indent=2),
            file_name=f"pathsnap_{st.session_state.get('trip', 'trip')}.json",
            mime="application/json", use_container_width=True,
        )
    with st.expander("Inspect raw API response"):
        st.json(result)
else:
    st.markdown('<div class="section">02 / Workspace</div>', unsafe_allow_html=True)
    ui.render_empty_state()

st.markdown(
    "<div class='footer-note'>PATHSNAP \u00b7 DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION \u00b7 "
    "Ground-truth labels remain server-side at all times</div>",
    unsafe_allow_html=True,
)
