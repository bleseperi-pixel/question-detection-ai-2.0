"""Flask web application for the College Question Predictor."""
from __future__ import annotations

import math

from flask import Flask, jsonify, render_template, request

from predictor import analyze

MAX_INPUT_CHARS = 100_000

app = Flask(__name__)


def _number(value, cast, low, high, default):
    """Convert a request value to a bounded number, falling back to default."""
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return default
    if isinstance(number, float) and not math.isfinite(number):
        return default
    return min(max(number, low), high)


def _parse_sections(raw_sections) -> list[dict]:
    if not isinstance(raw_sections, list):
        return []
    sections = []
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        marks = _number(item.get("marks"), int, 1, 20, 0)
        count = _number(item.get("count"), int, 0, 50, 0)
        if marks and count:
            sections.append({"marks": marks, "count": count})
    return sections


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}

    raw = str(payload.get("questions", "")).strip()
    if not raw:
        return jsonify(error="Paste the staff's questions first."), 400
    if len(raw) > MAX_INPUT_CHARS:
        return jsonify(error="Input is too large (maximum 100,000 characters)."), 413

    threshold = _number(payload.get("threshold"), float, 0.2, 0.95, 0.6)
    top_n = _number(payload.get("top_n"), int, 1, 50, 10)
    sections = _parse_sections(payload.get("sections"))
    title = str(payload.get("title") or "Question Paper").strip()[:120] or "Question Paper"
    subject = str(payload.get("subject") or "").strip()[:120]

    result = analyze(raw, threshold, top_n, sections or None, title, subject)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
