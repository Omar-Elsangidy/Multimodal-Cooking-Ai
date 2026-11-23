"""Phone/Tablet Web Dashboard (Flask) for Standalone YOLOv5s Kitchen Object Detection.

Streams annotated video and live object-detection telemetry over HTTP/LAN to any
phone, tablet, or desktop browser—with zero TAS or DAG dependencies.

Usage:
    python -m prototype.web_dashboard --source "C:\\Users\\ASUS\\Desktop\\P02_10.MP4" --stride 3 --host 0.0.0.0 --port 5000
    python -m prototype.web_dashboard --source 0 --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prototype.vision_detector import KitchenObjectDetector  # noqa: E402
from run_live_detection import open_capture, parse_timestamp  # noqa: E402

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multimodal Cooking Assistant — YOLO Live</title>
  <style>
    :root {
      --bg: #0f131a; --card: #181e28; --border: #2a3342; --text: #eef2f7;
      --muted: #9aa5b5; --accent: #38bdf8; --green: #22c55e;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    header h1 { margin: 0; font-size: 1.1rem; color: var(--accent); }
    .grid { display: grid; grid-template-columns: 1.45fr 1fr; gap: 14px; padding: 14px; max-width: 1320px; margin: 0 auto; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
    .card h2 { margin: 0 0 10px; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    img.stream { width: 100%; border-radius: 10px; display: block; background: #000; }
    .badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: #263142; }
    .badge.ok { background: rgba(34,197,94,0.2); color: var(--green); }
    .pill { display: inline-block; padding: 5px 10px; margin: 3px; border-radius: 999px; background: #232d3d; border: 1px solid #324056; font-size: 0.84rem; }
    .pill b { color: var(--accent); }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button, input[type=range] { background: #222b38; color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; }
    button { cursor: pointer; font-weight: 600; }
    button:hover { border-color: var(--accent); }
    .stat-row { display: flex; justify-content: space-between; font-size: 0.92rem; padding: 5px 0; border-bottom: 1px solid #222a36; }
    .stat-row:last-child { border-bottom: none; }
  </style>
</head>
<body>
  <header>
    <h1>Multimodal Cooking Assistant <span class="badge ok" id="modelBadge">YOLO</span></h1>
    <div class="controls">
      <button id="pauseBtn" onclick="togglePause()">Pause</button>
      <button onclick="restartVideo()">Restart</button>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Live Object Detection Stream</h2>
      <img class="stream" src="/video_feed" alt="Live YOLO Stream" />
      <div style="margin-top:10px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; font-size:0.9rem;">
        <span><strong>Time:</strong> <span id="timeText">00:00</span></span>
        <span><strong>Inference FPS:</strong> <span id="fpsText">0.0</span></span>
        <span><strong>Frame Boxes:</strong> <span id="boxCount">0</span></span>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:14px;">
      <div class="card">
        <h2>Confidence Threshold: <span id="confVal">0.25</span></h2>
        <input id="confSlider" type="range" min="0.10" max="0.80" step="0.05" value="0.25" style="width:100%" oninput="setConf(this.value)" />
      </div>

      <div class="card">
        <h2>Detected In Current Frame</h2>
        <div id="liveObjects"><span style="color:var(--muted)">Waiting for frames...</span></div>
      </div>

      <div class="card">
        <h2>Top Seen Objects (Session)</h2>
        <div id="topCounts"><span style="color:var(--muted)">No detections yet</span></div>
      </div>
    </div>
  </div>

  <script>
    async function setConf(val) {
      document.getElementById('confVal').textContent = parseFloat(val).toFixed(2);
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({conf: parseFloat(val)}) });
    }
    async function togglePause() {
      const res = await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({toggle_pause: true}) });
      const data = await res.json();
      document.getElementById('pauseBtn').textContent = data.paused ? 'Resume' : 'Pause';
    }
    async function restartVideo() {
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({restart: true}) });
    }
    async function poll() {
      try {
        const res = await fetch('/api/state');
        const s = await res.json();
        if (!s || !s.backend) return;

        document.getElementById('modelBadge').textContent = s.backend.toUpperCase() + ' (' + s.num_classes + ' cls)';
        const sec = Math.floor(s.timestamp_sec || 0);
        const mm = String(Math.floor(sec / 60)).padStart(2, '0');
        const ss = String(sec % 60).padStart(2, '0');
        document.getElementById('timeText').textContent = mm + ':' + ss;
        document.getElementById('fpsText').textContent = (s.fps || 0).toFixed(1);
        document.getElementById('boxCount').textContent = s.detections.length;
        document.getElementById('pauseBtn').textContent = s.paused ? 'Resume' : 'Pause';
        document.getElementById('confVal').textContent = (s.conf || 0.25).toFixed(2);
        document.getElementById('confSlider').value = s.conf || 0.25;

        document.getElementById('liveObjects').innerHTML = s.detections.length
          ? s.detections.map(d => `<span class="pill"><b>${d.label}</b> ${(d.conf * 100).toFixed(0)}%</span>`).join('')
          : '<span style="color:var(--muted)">none</span>';

        document.getElementById('topCounts').innerHTML = s.top_counts.length
          ? s.top_counts.map(([lbl, cnt]) => `<div class="stat-row"><span>${lbl}</span><strong>${cnt}</strong></div>`).join('')
          : '<span style="color:var(--muted)">No detections yet</span>';
      } catch (e) {}
    }
    setInterval(poll, 300);
    poll();
  </script>
</body>
</html>
"""


class DashboardRuntime:
    """Thread-safe video + YOLO worker serving frames and JSON state to Flask."""

    def __init__(self, detector: KitchenObjectDetector, source: str, start_sec: float, stride: int) -> None:
        self.detector = detector
        self.source = source
        self.start_sec = start_sec
        self.stride = max(1, stride)
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.class_counts: Counter = Counter()
        self.paused = False
        self.restart_requested = False
        self.stop_event = threading.Event()
        self.latest_state: Dict[str, Any] = {
            "backend": self.detector.backend,
            "num_classes": self.detector.num_classes,
            "timestamp_sec": 0.0,
            "fps": 0.0,
            "conf": round(self.detector.conf, 2),
            "paused": False,
            "detections": [],
            "top_counts": [],
        }

    def start(self) -> None:
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _loop(self) -> None:
        cap = open_capture(self.source, WORKSPACE_ROOT)
        if self.start_sec > 0 and not self.source.isdigit():
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = max(0.01, self.stride / src_fps)
        raw_idx = 0
        consecutive_misses = 0
        t0 = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if self.restart_requested and not self.source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)
                        self.class_counts.clear()
                        raw_idx = 0
                        consecutive_misses = 0
                        self.restart_requested = False
                    paused = self.paused

                if paused:
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok:
                    if not self.source.isdigit():
                        consecutive_misses += 1
                        target_ms = 0.0 if consecutive_misses > 1 else self.start_sec * 1000.0
                        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
                        raw_idx = 0
                        if consecutive_misses > 3:
                            time.sleep(0.05)
                        continue
                    break
                consecutive_misses = 0

                if raw_idx % self.stride != 0:
                    raw_idx += 1
                    continue
                raw_idx += 1

                pos_sec = (
                    (time.perf_counter() - t0)
                    if self.source.isdigit()
                    else ((cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0)
                )
                t_inf0 = time.perf_counter()
                with self.lock:
                    res = self.detector.detect(frame)
                    vis = self.detector.draw_boxes(frame, res.detections)
                    inf_fps = 1.0 / max(1e-6, time.perf_counter() - t_inf0)
                    for d in res.detections:
                        self.class_counts[d.label] += 1

                    ok_jpg, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok_jpg:
                        self.latest_jpeg = buf.tobytes()

                    det_list: List[Dict[str, Any]] = [
                        {"label": d.label, "conf": round(d.conf, 3)}
                        for d in sorted(res.detections, key=lambda x: x.conf, reverse=True)
                    ]
                    self.latest_state = {
                        "backend": self.detector.backend,
                        "num_classes": self.detector.num_classes,
                        "timestamp_sec": round(pos_sec, 2),
                        "fps": round(inf_fps, 1),
                        "conf": round(self.detector.conf, 2),
                        "paused": self.paused,
                        "detections": det_list,
                        "top_counts": self.class_counts.most_common(10),
                    }

                time.sleep(min(0.05, frame_delay * 0.25))
        finally:
            cap.release()


def create_app(runtime: DashboardRuntime) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/state")
    def api_state():
        with runtime.lock:
            return jsonify(runtime.latest_state)

    @app.route("/api/control", methods=["POST"])
    def api_control():
        data = request.get_json(silent=True) or {}
        with runtime.lock:
            if "conf" in data and data["conf"] is not None:
                runtime.detector.conf = max(0.05, min(0.95, float(data["conf"])))
            if data.get("toggle_pause"):
                runtime.paused = not runtime.paused
            if data.get("restart"):
                runtime.restart_requested = True
                runtime.paused = False
            paused = runtime.paused
        return jsonify({"ok": True, "paused": paused})

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
    parser = argparse.ArgumentParser(description="Flask Phone/Tablet UI for YOLO Object Detection")
    parser.add_argument("--source", type=str, required=True, help="Video path or webcam index ('0').")
    parser.add_argument("--detector", type=str, default="onnx", choices=["onnx", "world", "auto"])
    parser.add_argument("--onnx", type=str, default="exports/kitchen_yolov5s.onnx", help="ONNX model path.")
    parser.add_argument("--start", type=str, default="0", help="Start timestamp (seconds, MM:SS, or HH:MM:SS).")
    parser.add_argument("--stride", type=int, default=3, help="Process every Nth video frame.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (0.0.0.0 for LAN phone/tablet).")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port.")
    args = parser.parse_args()

    detector = KitchenObjectDetector(
        backend=args.detector,
        onnx_path=args.onnx,
        conf=args.conf,
        iou=args.iou,
    )
    runtime = DashboardRuntime(detector, args.source, parse_timestamp(args.start), args.stride)
    runtime.start()
    app = create_app(runtime)
    print(
        f"Web dashboard running at http://{args.host}:{args.port} "
        f"(backend={detector.backend}, classes={detector.num_classes})"
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
