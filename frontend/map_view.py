"""
Builds the Folium map for one prediction result.

Every layer here is drawn only from fields the backend actually returned -
nothing is invented, interpolated or hardcoded. The layer names mirror
API_CONTRACT.md so a reviewer can match map -> JSON field directly.

Road-network handling is graph-aware on purpose: PathSnap's raw .nodes/.arcs
files describe a GRAPH (nodes + the arcs connecting them), not a single
path (see the Dataset Description doc). Drawing an arbitrary list of road
points as one connected polyline - as a first-pass UI often does - draws a
line that does not exist in the data. This module instead:
  1. draws individual edges, if the backend sends edges directly or sends
     (nodes + arcs) in the raw dataset's own shape, and
  2. falls back to unconnected point markers, clearly labelled as such, if
     only a flat list of road points is available with no connectivity.
"""

import folium
from config import COLOR_GPS, COLOR_ROAD, COLOR_PREDICTED, COLOR_CLASSIFIER_ONLY, COLOR_REFERENCE


def _point(value):
    """Coerce one loosely-shaped coordinate into a (lat, lon) tuple, or None."""
    if isinstance(value, dict):
        lat = value.get("latitude", value.get("lat"))
        lon = value.get("longitude", value.get("lon", value.get("lng")))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        lat, lon = value[0], value[1]
    else:
        return None
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def _points(seq):
    out = []
    for item in seq or []:
        p = _point(item)
        if p is not None:
            out.append(p)
    return out


def _edges_from_pairs(seq):
    """Edges given directly as [[p1, p2], ...] or [{"from": p1, "to": p2}, ...]."""
    edges = []
    for item in seq or []:
        if isinstance(item, dict) and "from" in item and "to" in item:
            a, b = _point(item["from"]), _point(item["to"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = _point(item[0]), _point(item[1])
        else:
            a = b = None
        if a is not None and b is not None:
            edges.append((a, b))
    return edges


def _edges_from_nodes_and_arcs(nodes_raw, arcs_raw):
    """Edges given as the raw dataset's own .nodes + .arcs pair: node coordinates
    plus (from_node_id, to_node_id) index pairs."""
    nodes = _points(nodes_raw)
    edges = []
    for arc in arcs_raw or []:
        if isinstance(arc, dict):
            i, j = arc.get("from_node_id"), arc.get("to_node_id")
        elif isinstance(arc, (list, tuple)) and len(arc) == 2:
            i, j = arc[0], arc[1]
        else:
            continue
        try:
            a, b = nodes[int(i)], nodes[int(j)]
        except (TypeError, ValueError, IndexError):
            continue
        edges.append((a, b))
    return edges


def _road_edges(data):
    """Resolve whichever road-network shape the backend sent. Returns None if
    only a flat, connectivity-free point list is available."""
    if data.get("road_network_nodes") is not None or data.get("road_network_arcs") is not None:
        return _edges_from_nodes_and_arcs(data.get("road_network_nodes"), data.get("road_network_arcs"))
    raw = data.get("road_network") or []
    if raw and isinstance(raw[0], dict) and "from" in raw[0]:
        return _edges_from_pairs(raw)
    if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2 and _point(raw[0][0]) is not None:
        return _edges_from_pairs(raw)
    return None


def render_route_map(data, show_classifier_only=True, show_reference=False):
    """
    Build a Folium map from one /predict response's "map" object.
    Returns None (not an empty map) when nothing is plottable, so the caller
    can show an honest "no coordinates returned" message instead.
    """
    if not isinstance(data, dict):
        return None

    gps = _points(data.get("gps_points"))
    predicted = _points(data.get("predicted_route"))
    classifier_only = _points(data.get("classifier_only_route"))
    reference = _points(data.get("reference_route"))
    road_edges = _road_edges(data)
    road_points = [] if road_edges else _points(data.get("road_network"))

    all_points = (
        gps + predicted + classifier_only + reference + road_points
        + [pt for edge in (road_edges or []) for pt in edge]
    )
    if not all_points:
        return None

    center = [sum(p[0] for p in all_points) / len(all_points),
              sum(p[1] for p in all_points) / len(all_points)]
    # Plain OpenStreetMap tiles: no API key required. (Folium's "CartoDB
    # positron" preset now needs a Carto API key and silently fails without
    # one - not something you want discovered live during evaluation.)
    fmap = folium.Map(location=center, zoom_start=15, control_scale=True, tiles="OpenStreetMap")

    # Road network, drawn first so route layers sit visually on top of it.
    if road_edges:
        layer = folium.FeatureGroup(name=f"Road network ({len(road_edges)} segments)", show=True)
        for a, b in road_edges:
            folium.PolyLine([a, b], color=COLOR_ROAD, weight=2.5, opacity=0.65).add_to(layer)
        layer.add_to(fmap)
    elif road_points:
        layer = folium.FeatureGroup(
            name=f"Road-network nodes ({len(road_points)}, connectivity not supplied)", show=True
        )
        for p in road_points:
            folium.CircleMarker(p, radius=2, color=COLOR_ROAD, fill=True, fill_opacity=0.8, weight=0).add_to(layer)
        layer.add_to(fmap)

    # Raw GPS observations.
    if gps:
        layer = folium.FeatureGroup(name=f"GPS observations ({len(gps)})", show=True)
        stride = max(1, len(gps) // 2000)  # keep very long trips responsive in the browser
        for p in gps[::stride]:
            folium.CircleMarker(p, radius=2.5, color=COLOR_GPS, fill=True, fill_color=COLOR_GPS,
                                 fill_opacity=0.8, weight=1).add_to(layer)
        layer.add_to(fmap)

    # Reference / ground-truth route - hidden by default, and clearly marked.
    # Per the Dataset doc (s.7) and Models doc (s.6/s.11), evaluator ground truth
    # must stay server-side; this layer should only ever be populated for
    # student-testing trips, if the backend chooses to send it at all.
    if reference:
        layer = folium.FeatureGroup(
            name="Reference route (dev data only - must be empty for evaluator trips)",
            show=show_reference,
        )
        folium.PolyLine(reference, color=COLOR_REFERENCE, weight=4, opacity=0.85, dash_array="6,6").add_to(layer)
        layer.add_to(fmap)

    # Classifier-only (pre-HMM) prediction, if the backend exposes it - lets you
    # show live what the HMM/Viterbi decoder actually changed.
    if classifier_only:
        layer = folium.FeatureGroup(
            name="Classifier-only (before HMM / Viterbi)",
            show=show_classifier_only and bool(predicted),
        )
        folium.PolyLine(classifier_only, color=COLOR_CLASSIFIER_ONLY, weight=3.5,
                         opacity=0.75, dash_array="2,8").add_to(layer)
        layer.add_to(fmap)

    # Final predicted route - the headline result.
    if predicted:
        layer = folium.FeatureGroup(name="Predicted route (classifier + HMM/Viterbi)", show=True)
        folium.PolyLine(predicted, color=COLOR_PREDICTED, weight=6, opacity=0.95).add_to(layer)
        folium.Marker(predicted[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(layer)
        folium.Marker(predicted[-1], tooltip="End", icon=folium.Icon(color="red", icon="flag")).add_to(layer)
        layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(all_points)
    return fmap
