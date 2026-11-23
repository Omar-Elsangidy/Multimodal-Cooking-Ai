"""Phone/Tablet Front-End Web Dashboard (Flask) for the Multimodal Cooking Assistant.

Runs the full edge-AI pipeline (YOLO + TAS MS-TCN + Food Probe + CaptainCook4D DAG)
on the host/Raspberry Pi and streams the UI to any phone, tablet, or browser.

Usage:
    python -m prototype.web_dashboard --source "C:/Users/ASUS/Desktop/P02_10.MP4" --stride 3
    python -m prototype.web_dashboard --source 0 --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from prototype.pipeline import MultimodalCookingPipeline  # noqa: E402
from prototype.run_prototype import open_capture, parse_timestamp  # noqa: E402

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multimodal Cooking Assistant</title>
  <style>
    :root {
      --bg: #0f131a; --card: #181e28; --border: #2a3342; --text: #eef2f7;
      --muted: #9aa5b5; --accent: #38bdf8; --green: #22c55e; --warn: #f59e0b; --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    header h1 { margin: 0; font-size: 1.15rem; color: var(--accent); }
    .grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
    @media (max-width: 920px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
    .card h2 { margin: 0 0 10px; font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    img.stream { width: 100%; border-radius: 10px; display: block; background: #000; }
    .badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: #263142; }
    .badge.locked { background: rgba(34,197,94,0.2); color: var(--green); }
    .badge.action { background: rgba(56,189,248,0.18); color: var(--accent); font-size: 0.95rem; padding: 5px 12px; }
    .progress-wrap { width: 100%; height: 14px; background: #262f3d; border-radius: 999px; overflow: hidden; margin: 8px 0; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #22c55e, #38bdf8); transition: width 0.25s; }
    .alert { padding: 10px 12px; border-radius: 8px; margin-bottom: 10px; font-weight: 600; font-size: 0.9rem; }
    .alert.error { background: rgba(239,68,68,0.18); border: 1px solid var(--danger); color: #fca5a5; }
    .alert.temp { background: rgba(245,158,11,0.18); border: 1px solid var(--warn); color: #fde68a; }
    .pill { display: inline-block; padding: 4px 8px; margin: 2px; border-radius: 6px; background: #242d3b; font-size: 0.8rem; }
    ul.steps { list-style: none; padding: 0; margin: 0; }
    ul.steps li { padding: 7px 10px; border-radius: 8px; margin-bottom: 6px; background: #202734; font-size: 0.88rem; }
    ul.steps li.done { opacity: 0.7; border-left: 3px solid var(--green); }
    ul.steps li.next { border-left: 3px solid var(--accent); }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    select, button, input[type=range] { background: #222b38; color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; }
    button { cursor: pointer; font-weight: 600; }
    button:hover { border-color: var(--accent); }
  </style>
</head>
<body>
  <header>
    <h1>Multimodal Cooking Assistant <span class="badge" id="backendBadge">EDGE AI</span></h1>
    <div class="controls">
      <select id="recipeSelect" onchange="setRecipe(this.value)">
        <option value="">Auto-Detect Recipe</option>
        {% for rid, name in recipes %}
        <option value="{{ rid }}">{{ name }}</option>
        {% endfor %}
      </select>
      <button onclick="resetPipeline()">Reset Run</button>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Live Vision + Telemetry Feed</h2>
      <img class="stream" src="/video_feed" alt="Live Cooking Stream" />
      <div style="margin-top: 10px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div><strong>Detected Objects:</strong> <span id="objectsList">none</span></div>
        <div><strong>Action:</strong> <span class="badge action" id="actionBadge">IDLE (100%)</span></div>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:14px;">
      <div id="alertsBox"></div>

      <div class="card">
        <h2>Recipe State (CaptainCook4D DAG)</h2>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong id="recipeName" style="font-size:1.1rem;">Detecting...</strong>
          <span class="badge" id="lockBadge">CANDIDATE</span>
        </div>
        <div class="progress-wrap"><div class="progress-fill" id="progressBar" style="width:0%"></div></div>
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:var(--muted);">
          <span id="progressText">0% complete</span>
          <span id="candidatesText"></span>
        </div>
      </div>

      <div class="card">
        <h2>Food Probe Thermometer</h2>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <strong id="probeTemp" style="font-size:1.1rem; color:var(--accent);">22.0 C</strong>
          <span class="badge" id="probeLabel">Ambient</span>
        </div>
        <input id="tempSlider" type="range" min="20" max="220" value="22" style="width:100%" oninput="setTemp(this.value)" />
      </div>

      <div class="card">
        <h2>Next Valid Steps</h2>
        <ul class="steps" id="nextSteps"></ul>
      </div>

      <div class="card">
        <h2>Completed Steps</h2>
        <ul class="steps" id="doneSteps"></ul>
      </div>
    </div>
  </div>

  <script>
    async function setRecipe(rid) {
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({recipe_id: rid || null}) });
    }
    async function setTemp(val) {
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({probe_temp_c: parseFloat(val)}) });
    }
    async function resetPipeline() {
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({reset: true}) });
    }
    async function poll() {
      try {
        const res = await fetch('/api/state');
        const s = await res.json();
        if (!s || !s.selected_recipe_name) return;

        document.getElementById('backendBadge').textContent = 'YOLO: ' + s.detector_backend.toUpperCase();
        document.getElementById('recipeName').textContent = s.selected_recipe_name;
        const lockEl = document.getElementById('lockBadge');
        lockEl.textContent = s.recipe_locked ? 'LOCKED' : 'CANDIDATE (' + Math.round(s.recipe_confidence * 100) + '%)';
        lockEl.className = 'badge ' + (s.recipe_locked ? 'locked' : '');

        const pct = Math.round(s.completion_ratio * 100);
        document.getElementById('progressBar').style.width = pct + '%';
        document.getElementById('progressText').textContent = pct + '% complete (' + s.completed_steps.length + ' steps)';
        document.getElementById('candidatesText').textContent = 'Top-3: ' + s.top_k_candidates.map(c => c[0]).join(', ');

        document.getElementById('actionBadge').textContent = (s.tas_detail || s.tas_action).toUpperCase() + ' (' + Math.round(s.tas_confidence * 100) + '%)';
        document.getElementById('objectsList').innerHTML = s.detected_labels.length
          ? s.detected_labels.map(o => `<span class="pill">${o}</span>`).join('')
          : '<span class="muted">none</span>';

        document.getElementById('probeTemp').textContent = s.probe_temp_c.toFixed(1) + ' C' + (s.probe_target_c ? ` / target ${s.probe_target_c} C` : '');
        const probeBadge = document.getElementById('probeLabel');
        if (s.probe_target_c) {
          probeBadge.textContent = s.probe_gate_passed ? `SAFE >= ${s.probe_target_c} C` : `COOK < ${s.probe_target_c} C`;
          probeBadge.className = 'badge ' + (s.probe_gate_passed ? 'locked' : '');
        } else {
          probeBadge.textContent = s.probe_label;
          probeBadge.className = 'badge';
        }
        document.getElementById('tempSlider').value = Math.round(s.probe_temp_c);

        let alertsHtml = '';
        if (s.mistake_this_frame) alertsHtml += `<div class="alert error">ORDER ERROR: ${s.mistake_this_frame}</div>`;
        if (s.temp_alert) alertsHtml += `<div class="alert temp">${s.temp_alert}</div>`;
        document.getElementById('alertsBox').innerHTML = alertsHtml;

        document.getElementById('nextSteps').innerHTML = s.next_valid_steps.length
          ? s.next_valid_steps.map(([id, d]) => `<li class="next">[${id}] ${d}</li>`).join('')
          : '<li class="done">Recipe complete!</li>';

        document.getElementById('doneSteps').innerHTML = s.completed_steps.slice(-5).reverse()
          .map(([id, d]) => `<li class="done">[ok ${id}] ${d}</li>`).join('');
      } catch (e) {}
    }
    setInterval(poll, 350);
    poll();
  </script>
</body>
</html>
"""


class DashboardRuntime:
    """Thread-safe video + pipeline worker serving frames and JSON state to Flask."""

    def __init__(self, pipeline: MultimodalCookingPipeline, source: str, start_sec: float, stride: int) -> None:
        self.pipeline = pipeline
        self.source = source
        self.start_sec = start_sec
        self.stride = max(1, stride)
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.latest_state: Dict[str, Any] = {}
        self.stop_event = threading.Event()

    def start(self) -> None:
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _loop(self) -> None:
        cap = open_capture(self.source)
        if self.start_sec > 0 and not self.source.isdigit():
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = max(0.01, self.stride / src_fps)
        raw_idx = 0
        t0 = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not self.source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)
                        raw_idx = 0
                        continue
                    break

                if raw_idx % self.stride != 0:
                    raw_idx += 1
                    continue
                raw_idx += 1

                pos_sec = (time.perf_counter() - t0) if self.source.isdigit() else ((cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0)
                with self.lock:
                    state, hud_frame = self.pipeline.process_frame(frame, pos_sec)
                    ok_jpg, buf = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok_jpg:
                        self.latest_jpeg = buf.tobytes()
                    self.latest_state = state.to_dict()

                time.sleep(min(0.05, frame_delay * 0.25))
        finally:
            cap.release()


def create_app(runtime: DashboardRuntime) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        recipes = sorted((rid, dag.recipe_name) for rid, dag in runtime.pipeline.recipes.items())
        return render_template_string(HTML_TEMPLATE, recipes=recipes)

    @app.route("/api/state")
    def api_state():
        with runtime.lock:
            return jsonify(runtime.latest_state)

    @app.route("/api/control", methods=["POST"])
    def api_control():
        data = request.get_json(silent=True) or {}
        with runtime.lock:
            if data.get("reset"):
                runtime.pipeline.reset()
            if "recipe_id" in data:
                runtime.pipeline.set_recipe(data["recipe_id"])
            if "probe_temp_c" in data and data["probe_temp_c"] is not None:
                runtime.pipeline.temp_monitor.set_temp(float(data["probe_temp_c"]))
            if "probe_mode" in data:
                runtime.pipeline.temp_monitor.set_mode(str(data["probe_mode"]))
        return jsonify({"ok": True})

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while not runtime.stop_event.is_set():
                with runtime.lock:
                    jpg = runtime.latest_jpeg
                if jpg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.04)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Flask Phone/Tablet UI for Multimodal Cooking Assistant")
    parser.add_argument("--source", type=str, required=True, help="Video path or webcam index ('0').")
    parser.add_argument("--detector", type=str, default="auto", choices=["auto", "world", "onnx"])
    parser.add_argument("--recipe", type=str, default=None, help="Optional initial recipe ID.")
    parser.add_argument("--start", type=str, default="0", help="Start timestamp (seconds or MM:SS).")
    parser.add_argument("--stride", type=int, default=3, help="Process every Nth video frame.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (0.0.0.0 for LAN phone/tablet).")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port.")
    args = parser.parse_args()

    pipeline = MultimodalCookingPipeline(
        detector_backend=args.detector,
        forced_recipe_id=args.recipe,
        detect_every_n=1,
    )
    runtime = DashboardRuntime(pipeline, args.source, parse_timestamp(args.start), args.stride)
    runtime.start()
    app = create_app(runtime)
    print(f"Web dashboard running at http://{args.host}:{args.port} (detector={pipeline.detector.backend})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
