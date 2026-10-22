# -*- coding: utf-8 -*-
"""
Flask web application — CRISP-DM Multiple Linear Regression Studio.

Hosts several "problems" (Kaggle datasets analysed with CRISP-DM + OLS):

  startups  50 Startups                     (5 features,  model_data.json)
  house     House Sales in King County USA  (16 features, house_model_data.json)

Endpoints
  GET  /                       interactive dashboard (index.html, shared by all problems)
  GET  /api/problems           which problems are available
  GET  /api/models?problem=id  full precomputed payload {meta, models} (or ?algo=)
  GET  /api/summary?problem=id flat list of every (method, k) evaluation
  POST /api/predict            live inference from the stored OLS coefficients
                               {"problem":"house","algo":"Forward SFS","k":8,
                                "values":{"sqft_living":1910,...}}
  GET  /api/dataset?url=...    server-side CSV fetch (CORS helper for the
                               "add any dataset" dialog; the analysis itself
                               always runs in the browser)

Run
  pip install -r requirements.txt
  python app.py         ->  http://localhost:5000/
"""

import json
import os
import urllib.request
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_DATASET_BYTES = 20 * 1024 * 1024       # 20 MB, mirrors static/analyze.js

PROBLEM_FILES = {
    "startups": "model_data.json",
    "house": "house_model_data.json",
}

app = Flask(__name__, static_folder="static")
app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:                                # pragma: no cover
        print(f"Warning: could not load {os.path.basename(path)} ({exc}).")
        return {"meta": {}, "models": {}}


# every problem is loaded once at start-up
PAYLOADS = {}
for _pid, _fname in PROBLEM_FILES.items():
    _p = os.path.join(BASE_DIR, _fname)
    if os.path.exists(_p):
        PAYLOADS[_pid] = _load(_p)
if not PAYLOADS:                                            # pragma: no cover
    PAYLOADS["startups"] = {"meta": {}, "models": {}}

DEFAULT_PROBLEM = "startups" if "startups" in PAYLOADS else next(iter(PAYLOADS))


def payload_for():
    """Resolve ?problem= for GET endpoints (defaults to the first available problem)."""
    pid = request.args.get("problem") or DEFAULT_PROBLEM
    return pid, PAYLOADS.get(pid)


def _unknown(pid):
    return jsonify({
        "status": "error",
        "message": f"unknown problem '{pid}'",
        "async_analysis": "datasets added in the browser are not stored server side",
        "available": list(PAYLOADS),
    }), 404


@app.route("/")
def index():
    """Serve the dashboard (the same static file GitHub Pages publishes)."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/problems", methods=["GET"])
def api_problems():
    return jsonify([
        {
            "id": pid,
            "name": payload.get("meta", {}).get("dataset", {}).get("name", pid),
            "features": len(payload.get("meta", {}).get("dataset", {}).get("features", [])),
            "methods": list(payload.get("models", {}).keys()),
        }
        for pid, payload in PAYLOADS.items()
    ])


@app.route("/api/models", methods=["GET"])
def api_models():
    """Full payload, or a single method when ?algo= is given."""
    pid, payload = payload_for()
    if payload is None:
        return _unknown(pid)
    models = payload.get("models", {})
    algo = request.args.get("algo")
    if algo:
        if algo not in models:
            return jsonify({"error": f"unknown algorithm '{algo}'"}), 404
        return jsonify({"problem": pid, "meta": payload.get("meta", {}), "models": {algo: models[algo]}})
    return jsonify(payload)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """One flat row per (method, k) so clients can compare everything at once."""
    pid, payload = payload_for()
    if payload is None:
        return _unknown(pid)
    rows = []
    for algo, entries in payload.get("models", {}).items():
        for entry in entries:
            rows.append({
                "problem": pid,
                "algorithm": algo,
                "k": entry["k"],
                "features": entry["features"],
                "rmse": entry["rmse"],
                "mae": entry["mae"],
                "r2": entry["r2"],
                "adj_r2": entry["adj_r2"],
                "cv_rmse": entry["cv_rmse"],
            })
    return jsonify(rows)


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    """Fetch a CSV server-side so the browser can analyse hosts without CORS headers."""
    url = (request.args.get("url") or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return jsonify({"status": "error", "message": "url must be an absolute http(s) URL"}), 400
    host = parsed.hostname.lower()
    if host in ("localhost", "0.0.0.0", "::1") or host.startswith(
            ("127.", "10.", "192.168.", "169.254.", "172.16.", "172.17.", "172.18.", "172.19.")) \
            or host.endswith(".local"):
        return jsonify({"status": "error", "message": "private hosts are not allowed"}), 400
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "crispdm-mlr-studio/1.0 (+https://github.com/)",
            "Accept": "text/csv,text/plain,*/*",
        })
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read(MAX_DATASET_BYTES + 1)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 502
    if len(raw) > MAX_DATASET_BYTES:
        return jsonify({"status": "error", "message": "file is larger than 20 MB"}), 413
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return jsonify({"status": "success", "url": url, "bytes": len(raw), "csv": text})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Score one observation from the precomputed coefficients of (problem, algo, k)."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        pid = data.get("problem") or DEFAULT_PROBLEM
        if pid not in PAYLOADS:
            return jsonify({"status": "error", "message": f"unknown problem '{pid}'"}), 400
        payload = PAYLOADS[pid]
        models = payload.get("models", {})
        meta = payload.get("meta", {})
        specs = meta.get("predictor_inputs", [])

        algo = data.get("algo") or next(iter(models), None)
        if algo not in models:
            return jsonify({"status": "error", "message": f"unknown algorithm '{algo}'"}), 400
        k = int(data.get("k", 2))
        if not 1 <= k <= len(models[algo]):
            return jsonify({"status": "error", "message": f"k must be 1..{len(models[algo])}"}), 400

        given = data.get("values") or {}
        # legacy shape (problem 1 only): rd_spend / marketing_spend / admin_spend / state
        if not given and "rd_spend" in data:
            given = {
                "rd": data.get("rd_spend"),
                "admin": data.get("admin_spend"),
                "marketing": data.get("marketing_spend"),
                "state": data.get("state"),
            }
        values = {}
        for spec in specs:
            raw = given.get(spec["key"], spec.get("default"))
            if spec.get("type") == "select":
                if raw is None:
                    raw = spec.get("default")
                values[spec["key"]] = str(raw)
            else:
                values[spec["key"]] = float(spec.get("default") if raw is None else raw)

        def x_for(name):
            if name.startswith("Intercept"):
                return 1.0
            for spec in specs:
                if spec.get("type") == "range" and spec.get("feature") == name:
                    return values.get(spec["key"], 0.0)
                if spec.get("type") == "select" and name.startswith(spec.get("prefix", "\u0000")):
                    return 1.0 if values.get(spec["key"]) == name[len(spec["prefix"]):] else 0.0
            return 0.0

        spec = models[algo][k - 1]
        predicted, breakdown = 0.0, []
        for coef in spec["coefficients"]:
            x = x_for(coef["feature"])
            contribution = coef["coef"] * x
            predicted += contribution
            breakdown.append({
                "feature": coef["feature"],
                "value": x,
                "coef": coef["coef"],
                "contribution": contribution,
            })

        return jsonify({
            "status": "success",
            "problem": pid,
            "algorithm": algo,
            "k": k,
            "features": spec["features"],
            "equation": spec["equation"],
            "prediction": predicted,
            "predicted_profit": predicted,          # backwards compatible (problem 1)
            "target": meta.get("dataset", {}).get("target"),
            "rmse": spec["rmse"],
            "r2": spec["r2"],
            "breakdown": breakdown,
        })
    except Exception as exc:                               # pragma: no cover
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" * CRISP-DM MLR studio -> http://localhost:{port}/   (problems: {', '.join(PAYLOADS)})")
    app.run(host="0.0.0.0", port=port, debug=False)
