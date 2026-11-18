import argparse
import socket
import sys
import time
import json
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, log_loss

K_CAND = 10                          
EXPOSE_REFERENCE_FOR_DEV_TRIPS = True    
ROAD_PAD_M = (300, 150, 50)               
MAX_ROAD_ARCS = 20000
LOG_FILE = Path("pathsnap_api.log")
SCRIPT_DIR = Path(__file__).resolve().parent

api_model = api_features = api_hmm = None
api_model_name = ""
TRIP_DIRS = {}
SERVED_PARTITION = ""


def project_xy(lon, lat, lon0, lat0):
    x = np.radians(np.asarray(lon) - lon0) * 6371000.0 * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * 6371000.0
    return np.column_stack([x, y])


def segment_distance(P, A, B):
    V = B - A
    den = (V ** 2).sum(-1)
    den = np.where(den == 0, 1e-9, den)
    t = np.clip(((P - A) * V).sum(-1) / den, 0, 1)
    return np.linalg.norm(P - (A + t[..., None] * V), axis=-1)


def angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def trip_geometry(data):
    tr = data["track"]
    nodes = data["nodes"].to_numpy(float)
    arcs = data["arcs"].to_numpy(int)
    lon0, lat0 = tr["lon"].mean(), tr["lat"].mean()
    nxy = project_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    return {"P": project_xy(tr["lon"], tr["lat"], lon0, lat0),
            "S": nxy[arcs[:, 0]], "E": nxy[arcs[:, 1]], "arcs": arcs}


def arc_sample_points(S, E, step=25.0):
    n = np.maximum(2, np.ceil(np.linalg.norm(E - S, axis=1) / step).astype(int) + 1)
    arc = np.repeat(np.arange(len(S)), n)
    first = np.cumsum(n) - n
    frac = (np.arange(n.sum()) - np.repeat(first, n)) / np.repeat(n - 1, n)
    return S[arc] + frac[:, None] * (E[arc] - S[arc]), arc


def generate_candidates(data, geom, k=K_CAND, probe=40):
    # Candidate roads + features for every GPS point. Uses NO route / label information.
    P, S, E, arcs = geom["P"], geom["S"], geom["E"], geom["arcs"]
    tr = data["track"]
    n = len(P)
    pts, arc_of_pt = arc_sample_points(S, E)
    _, idx = cKDTree(pts).query(P, k=min(probe, len(pts)))
    cand = np.sort(arc_of_pt[idx], axis=1)
    dup = np.zeros(cand.shape, bool)
    dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
    dist = segment_distance(P[:, None, :], S[cand], E[cand])
    dist[dup] = np.inf                                   # an arc is kept only once per GPS point
    kk = min(k, cand.shape[1])
    order = np.argsort(dist, axis=1)[:, :kk]
    cand = np.take_along_axis(cand, order, 1)
    dist = np.take_along_axis(dist, order, 1)
    ok = np.isfinite(dist)

    t = tr["time"].to_numpy(float)
    prev = np.vstack([P[:1], P[:-1]])
    nxt = np.vstack([P[1:], P[-1:]])
    dt = np.maximum(np.diff(t, prepend=t[0]), 1.0)
    speed = np.linalg.norm(P - prev, axis=1) / dt
    move = nxt - prev
    moving = np.linalg.norm(move, axis=1) > 1.0
    gps_bearing = np.arctan2(move[:, 0], move[:, 1])
    seg_bearing = np.arctan2(E[:, 0] - S[:, 0], E[:, 1] - S[:, 1])
    ddiff = angle_diff(gps_bearing[:, None], seg_bearing[cand])
    ddiff[~moving, :] = np.pi / 2                       # heading unknown -> neutral value
    mid = (S + E) / 2
    seglen = np.linalg.norm(E - S, axis=1)
    obs = np.repeat(np.arange(n), kk).reshape(n, kk)
    return pd.DataFrame({
        "trip_id": data["trip_id"], "obs_index": obs[ok], "time": t[obs[ok]], "arc_id": cand[ok],
        "from_node": arcs[cand[ok], 0], "to_node": arcs[cand[ok], 1],
        "distance_m": dist[ok],
        "midpoint_distance_m": np.linalg.norm(P[:, None, :] - mid[cand], axis=-1)[ok],
        "segment_length_m": seglen[cand][ok],
        "speed_mps": np.repeat(speed, kk).reshape(n, kk)[ok],
        "direction_difference": ddiff[ok],
        "distance_rank": np.tile(np.arange(kk), (n, 1))[ok],
        "relative_distance_m": (dist - dist[:, :1])[ok]})


def reference_arcs(data, geom):
    # Approximate reference label: nearest arc among the supplied route arcs (metrics only, never returned for evaluator trips).
    route = np.unique(np.array(data["route"], dtype=int))
    d = segment_distance(geom["P"][:, None, :], geom["S"][route][None], geom["E"][route][None])
    return pd.DataFrame({"obs_index": np.arange(len(geom["P"])), "ref_arc": route[d.argmin(1)],
                         "ref_distance_m": d.min(1)})


def add_labels(cands, ref):
    out = cands.merge(ref[["obs_index", "ref_arc"]], on="obs_index", how="left")
    out["is_match"] = (out["arc_id"] == out["ref_arc"]).astype(int)
    return out


def top1_and_miss(frame, score):
    tmp = frame[["trip_id", "obs_index", "is_match"]].copy()
    tmp["score"] = np.asarray(score)
    best = tmp.loc[tmp.groupby(["trip_id", "obs_index"])["score"].idxmax()]
    has_ref = tmp.groupby(["trip_id", "obs_index"])["is_match"].max()
    return float(best["is_match"].mean()), float((has_ref == 0).mean())


def viterbi_decode(cands, scores, arcs, max_hops=3, penalty=-4.0, gap_s=10.0):
    # cands: candidate rows of ONE trip; scores: classifier probabilities (same row order); arcs: array [n_arcs, 2].
    f = cands[["obs_index", "time", "arc_id"]].copy()
    f["p"] = np.clip(np.asarray(scores), 1e-6, 1.0)
    f["p"] = f["p"] / f.groupby("obs_index")["p"].transform("sum")
    groups = [g for _, g in f.groupby("obs_index", sort=True)]
    if not groups:
        return pd.DataFrame(columns=["obs_index", "arc_id"])

    out_arcs = {}
    for a, (u, v) in enumerate(arcs):
        out_arcs.setdefault(int(u), []).append(a)
    reach_cache = {}

    def reachable(a):
        if a not in reach_cache:
            seen, frontier = {a}, [a]
            for _ in range(max_hops):
                nxt = []
                for x in frontier:
                    for y in out_arcs.get(int(arcs[x][1]), []):
                        if y not in seen:
                            seen.add(y)
                            nxt.append(y)
                frontier = nxt
            reach_cache[a] = seen
        return reach_cache[a]

    prev = groups[0]
    delta = np.log(prev["p"].to_numpy())
    back = []
    for g in groups[1:]:
        gap = float(g["time"].iloc[0] - prev["time"].iloc[0]) > gap_s
        prev_arcs, cur_arcs = prev["arc_id"].to_numpy(), g["arc_id"].to_numpy()
        T = np.zeros((len(prev_arcs), len(cur_arcs)))
        if not gap:
            for i, a in enumerate(prev_arcs):
                reach = reachable(int(a))
                for j, b in enumerate(cur_arcs):
                    if int(b) not in reach:
                        T[i, j] = penalty
        total = delta[:, None] + T
        best_prev = total.argmax(0)
        delta = total[best_prev, np.arange(len(cur_arcs))] + np.log(g["p"].to_numpy())
        back.append(best_prev)
        prev = g
    s = int(delta.argmax())
    path = [s]
    for b in reversed(back):
        s = int(b[s])
        path.append(s)
    path = path[::-1]
    return pd.DataFrame({"obs_index": [int(g["obs_index"].iloc[0]) for g in groups],
                         "arc_id": [int(g["arc_id"].iloc[i]) for g, i in zip(groups, path)]})


def resolve(p):
    """Use the path as given; if it does not exist, try it relative to this script's folder."""
    p = Path(p).expanduser()
    return p if p.exists() else (SCRIPT_DIR / p)


def discover_trips(root):
    """{trip_id: folder} for every <id>/<id>.track under root."""
    return {f.parent.name: f.parent for f in sorted(Path(root).rglob("*.track"))
            if f.stem == f.parent.name and "__MACOSX" not in f.parts}


def find_evaluator_dir(data_dir):
    """Look inside data_dir, then at the siblings of data_dir (one level only - never scans .venv etc.)."""
    inside = [p for p in data_dir.rglob("*")
              if p.is_dir() and "evaluator" in p.name.lower() and "__MACOSX" not in p.parts]
    if inside:
        return inside[0]
    beside = sorted(p for p in data_dir.parent.iterdir()
                    if p.is_dir() and "evaluator" in p.name.lower() and "__MACOSX" not in p.parts)
    return beside[0] if beside else None


def find_student_test_dir(data_dir):
    dirs = [p for p in data_dir.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts]
    test = [p for p in dirs if "test" in p.name.lower() and "evaluator" not in p.name.lower()]
    return test[0] if test else None


API_REQUEST_LOG = deque(maxlen=500)


def api_log(event, **info):
    row = {"time": datetime.now().strftime("%H:%M:%S"), "event": event, **info}
    API_REQUEST_LOG.append(row)
    line = "[PathSnap API] " + " | ".join(f"{k}={v}" for k, v in row.items())
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")
    except OSError:
        pass


def init(model_dir, data_dir, evaluator_dir=None):
    """Load the 3 artifacts and build the server-side trip registry."""
    global api_model, api_features, api_hmm, api_model_name, TRIP_DIRS, SERVED_PARTITION

    model_dir, data_dir = resolve(model_dir), resolve(data_dir)
    for f in ("pathsnap_model.joblib", "feature_columns.json", "hmm_config.json"):
        if not (model_dir / f).exists():
            raise FileNotFoundError(f"Missing {model_dir / f} - run the notebook's 'save artifacts' section once.")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()} (current dir: {Path.cwd()})")

    api_model = joblib.load(model_dir / "pathsnap_model.joblib")
    api_features = json.loads((model_dir / "feature_columns.json").read_text())
    api_hmm = json.loads((model_dir / "hmm_config.json").read_text())
    last = api_model[-1] if hasattr(api_model, "__getitem__") else api_model
    api_model_name = type(last).__name__
    print("Loaded model:", api_model_name, "| features:", api_features, "| HMM:", api_hmm)

    ev_dir = resolve(evaluator_dir) if evaluator_dir else find_evaluator_dir(data_dir)
    eval_trips = discover_trips(ev_dir) if ev_dir and Path(ev_dir).exists() else {}
    if eval_trips:
        SERVED_PARTITION, TRIP_DIRS = "evaluator", eval_trips
        print("Evaluator folder:", ev_dir)
    else:
        st_dir = find_student_test_dir(data_dir)
        if st_dir is None:
            raise FileNotFoundError("No evaluator trips and no student-test folder found.")
        SERVED_PARTITION, TRIP_DIRS = "student_test", discover_trips(st_dir)
        print("WARNING: no evaluator trips found - serving the STUDENT-TEST trips instead (fine for development).")
    print(f"Serving {len(TRIP_DIRS)} '{SERVED_PARTITION}' trips:", sorted(TRIP_DIRS))


def api_read_trip(td):
    """Reads a trip folder; the .route file is OPTIONAL (evaluator trips may not ship one)."""
    tid = td.name
    raw = pd.read_csv(td / f"{tid}.track", sep=r"\s+", header=None, names=["lon", "lat", "time"])
    nodes = pd.read_csv(td / f"{tid}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(td / f"{tid}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    try:
        route = pd.read_csv(td / f"{tid}.route", sep=r"\s+", header=None, names=["arc_id"])["arc_id"].astype(int).tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        route = []
    track = raw.replace([np.inf, -np.inf], np.nan).dropna()
    track = track[track["lon"].between(-180, 180) & track["lat"].between(-90, 90)]
    track = track.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return {"trip_id": tid, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def api_route_continuity(seq, arcs):
    seq = [int(a) for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if len(seq) < 2:
        return None
    return sum(int(arcs[a][1]) == int(arcs[b][0]) for a, b in zip(seq[:-1], seq[1:])) / (len(seq) - 1)


def arcs_to_polyline(arc_seq, nodes, arcs):
    """Arc ids -> [[lat, lon], ...]. Consecutive repeats are collapsed; a connected arc adds only its end node."""
    pts, last, prev = [], None, None
    for a in arc_seq:
        a = int(a)
        if a == prev:
            continue
        prev = a
        u, v = int(arcs[a][0]), int(arcs[a][1])
        if last != u:
            pts.append([float(nodes[u, 1]), float(nodes[u, 0])])
        pts.append([float(nodes[v, 1]), float(nodes[v, 0])])
        last = v
    return pts


def road_network_payload(geom, nodes):
    """Nodes as [lat, lon] + arcs as [from_idx, to_idx], clipped to the neighbourhood of the trip."""
    P, S, E, arcs = geom["P"], geom["S"], geom["E"], geom["arcs"]
    for pad in ROAD_PAD_M:
        lo, hi = P.min(0) - pad, P.max(0) + pad
        inside = ((S > lo) & (S < hi)).all(1) | ((E > lo) & (E < hi)).all(1)
        if inside.sum() <= MAX_ROAD_ARCS:
            break
    sel = arcs[inside]
    used, inv = np.unique(sel.ravel(), return_inverse=True)
    return ([[float(nodes[n, 1]), float(nodes[n, 0])] for n in used],
            inv.reshape(-1, 2).tolist())


def _num(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def run_pathsnap(trip_id):
    """Candidates -> classifier -> Viterbi -> JSON-ready dict (see API_CONTRACT.md)."""
    info = TRIP_DIRS[trip_id]
    data = api_read_trip(info)
    geom = trip_geometry(data)
    arcs, nodes, tr = geom["arcs"], data["nodes"].to_numpy(float), data["track"]

    cands = generate_candidates(data, geom)                      # uses NO route / label information
    if cands.empty:
        raise ValueError("no candidate roads were generated for this trip")

    t0 = time.perf_counter()
    cands = cands.assign(score=api_model.predict_proba(cands[api_features])[:, 1])
    cands = cands.sort_values(["obs_index", "distance_rank"]).reset_index(drop=True)
    clf = cands.loc[cands.groupby("obs_index")["score"].idxmax()].set_index("obs_index")["arc_id"]
    hmm = viterbi_decode(cands, cands["score"].to_numpy(), arcs, **api_hmm).set_index("obs_index")["arc_id"]
    infer_ms = (time.perf_counter() - t0) * 1000

    hmm_arcs, clf_arcs = hmm.tolist(), clf.tolist()
    metrics = {"route_continuity": _num(api_route_continuity(hmm_arcs, arcs)),
               "classifier_route_continuity": _num(api_route_continuity(clf_arcs, arcs)),
               "inference_time_ms": round(infer_ms, 1)}

    # Metrics need labels. They are computed server-side only if a .route file exists; the labels are never returned.
    has_ref = len(data["route"]) > 0
    if has_ref:
        ref = reference_arcs(data, geom)
        ref_s = ref.set_index("obs_index")["ref_arc"]
        lab = add_labels(cands, ref)
        pred = (lab["score"] >= 0.5).astype(int)
        _, miss = top1_and_miss(lab, lab["score"])
        metrics.update({
            "top1_accuracy": _num((hmm.reindex(ref_s.index) == ref_s).mean()),
            "classifier_top1_accuracy": _num((clf.reindex(ref_s.index) == ref_s).mean()),
            "macro_f1": _num(f1_score(lab["is_match"], pred, average="macro")),
            "log_loss": _num(log_loss(lab["is_match"], np.clip(lab["score"], 1e-6, 1 - 1e-6), labels=[0, 1])),
            "candidate_miss_rate": _num(miss)})

    road_nodes, road_arcs = road_network_payload(geom, nodes)
    hmm_line = arcs_to_polyline(hmm_arcs, nodes, arcs)
    map_obj = {"gps_points": tr[["lat", "lon"]].to_numpy().tolist(),
               "predicted_route": hmm_line,
               "classifier_only_route": arcs_to_polyline(clf_arcs, nodes, arcs),
               "road_network_nodes": road_nodes, "road_network_arcs": road_arcs}
    if has_ref and EXPOSE_REFERENCE_FOR_DEV_TRIPS and SERVED_PARTITION == "student_test":
        map_obj["reference_route"] = arcs_to_polyline(data["route"], nodes, arcs)

    return {"trip_id": trip_id,
            "partition": SERVED_PARTITION,
            "summary": {"observation_count": int(len(tr)),
                        "predicted_segment_count": int(len(list(dict.fromkeys(
                            a for i, a in enumerate(hmm_arcs) if i == 0 or a != hmm_arcs[i - 1])))),
                        "candidate_rows": int(len(cands)),
                        "model": api_model_name, "hmm_config": api_hmm,
                        "reference_available_server_side": has_ref},
            "metrics": metrics,
            "map": map_obj}


app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": api_model_name, "partition": SERVED_PARTITION, "trips_available": len(TRIP_DIRS)}


@app.get("/trips")
def trips():
    return {"trip_ids": sorted(TRIP_DIRS), "partition": SERVED_PARTITION}


@app.get("/logs")
def logs(n: int = 50):
    """Last n request-log rows (handy to show the evaluator in a browser)."""
    return list(API_REQUEST_LOG)[-max(1, min(n, 500)):]


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    client = request.client.host if request.client else "?"
    if tid not in TRIP_DIRS:                                      # whitelist lookup: no client-supplied paths
        api_log("predict", trip=tid, client=client, status=404)
        raise HTTPException(404, f"Unknown trip_id '{tid}'. GET /trips lists the available ids.")
    t0 = time.perf_counter()
    try:
        out = run_pathsnap(tid)
    except Exception as exc:
        traceback.print_exc()
        api_log("predict", trip=tid, client=client, status=500, error=repr(exc))
        raise HTTPException(500, f"Prediction failed for '{tid}': {exc}")
    ms = (time.perf_counter() - t0) * 1000
    api_log("predict", trip=tid, client=client, status=200, ms=round(ms), obs=out["summary"]["observation_count"],
            top1=out["metrics"].get("top1_accuracy"), continuity=out["metrics"].get("route_continuity"))
    return out


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]                            # no packet is actually sent
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="PathSnap API server")
    ap.add_argument("--model-dir", default="pathsnap_output-1", help="folder with the 3 saved artifacts")
    ap.add_argument("--data-dir", default="Student folder", help="extracted dataset folder")
    ap.add_argument("--evaluator-dir", default=None, help="evaluator trips folder (auto-searched if omitted)")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from the other laptop")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    init(args.model_dir, args.data_dir, args.evaluator_dir)
    print(f"\nPathSnap API starting.\n  this laptop : http://127.0.0.1:{args.port}\n"
          f"  other laptop: http://{lan_ip()}:{args.port}   <- paste this into the frontend sidebar\n"
          f"  docs        : http://127.0.0.1:{args.port}/docs\n"
          f"  live log    : http://127.0.0.1:{args.port}/logs   (also written to {LOG_FILE})\n"
          f"  stop        : Ctrl+C\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()