"""
Web Application for the AI Science Discovery Team.
Provides a real-time dashboard to monitor and control the discovery pipeline.

This dashboard visualizes the "thinking process" of the AI team, allowing the user
to see the "Step Decomposer" break down problems and the "Physics Oracle" validate them
in real-time. This transparency is key to trusting the "First Principles" approach.
"""

import os
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pathlib import Path

from config import LM_STUDIO_BASE_URL, RESULTS_DIR
from pipeline import DiscoveryPipeline
from llm_client import LLMClient

app = Flask(__name__)
app.config["SECRET_KEY"] = "science-discovery-2026"
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

# Global state
pipeline_instance = None
pipeline_thread = None
log_history = []
current_progress = {"step": 0, "total": 10, "message": "Idle"}

# Persistent log file path
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline_log.txt")


def log_callback(message):
    """Send log messages to all connected clients AND save to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"time": timestamp, "message": message}
    log_history.append(entry)
    # Keep last 500 log entries in memory
    if len(log_history) > 500:
        log_history.pop(0)
    socketio.emit("log", entry)

    # ── Always persist to file so nothing is ever lost ──
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass  # Don't crash the pipeline over a log write failure


def progress_callback(step, total, message):
    """Send progress updates to all connected clients."""
    global current_progress
    current_progress = {"step": step, "total": total, "message": message}
    socketio.emit("progress", current_progress)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Get current system status."""
    client = LLMClient()
    connected, models = client.check_connection()
    loaded_model = client.get_loaded_model()

    is_running = pipeline_instance is not None and pipeline_instance.is_running

    return jsonify({
        "lm_studio_connected": connected,
        "available_models": models,
        "loaded_model": loaded_model,
        "pipeline_running": is_running,
        "progress": current_progress,
        "log_count": len(log_history),
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start the discovery pipeline."""
    global pipeline_instance, pipeline_thread

    if pipeline_instance and pipeline_instance.is_running:
        return jsonify({"error": "Pipeline is already running"}), 400

    data = request.json
    problem = data.get("problem", "").strip()
    if not problem:
        return jsonify({"error": "No problem description provided"}), 400

    # Clear previous logs
    log_history.clear()

    pipeline_instance = DiscoveryPipeline(
        log_callback=log_callback,
        progress_callback=progress_callback,
    )

    def run_pipeline():
        try:
            log_callback("🚀 Starting AI Science Discovery Pipeline...")
            log_callback(f"📋 Problem: {problem[:200]}...")
            log_callback(f"📁 Logs are being saved to: {os.path.abspath(LOG_FILE)}")
            pipeline_instance.run(problem)
            log_callback("🏁 Pipeline complete!")
            progress_callback(10, 10, "✅ Complete!")
        except Exception as e:
            log_callback(f"❌ Pipeline crashed: {e}")
            import traceback
            log_callback(traceback.format_exc())

    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the discovery pipeline."""
    global pipeline_instance
    if pipeline_instance and pipeline_instance.is_running:
        pipeline_instance.stop()
        return jsonify({"status": "stopping"})
    return jsonify({"error": "Pipeline is not running"}), 400

@app.route("/api/hard_reset", methods=["POST"])
def api_hard_reset():
    """Forcefully reset the pipeline instance."""
    global pipeline_instance
    if pipeline_instance:
        pipeline_instance.stop()
        # We can't easily kill a python thread, but setting the flag
        # will stop it after the current LLM chunk.
    pipeline_instance = None
    return jsonify({"status": "reset_complete"})


@app.route("/api/results")
def api_results():
    """Get past discovery results."""
    results = []
    results_dir = RESULTS_DIR
    if os.path.exists(results_dir):
        for f in sorted(os.listdir(results_dir), reverse=True):
            if f.startswith("summary_") and f.endswith(".md"):
                filepath = os.path.join(results_dir, f)
                with open(filepath, "r", encoding="utf-8") as fh:
                    content = fh.read()
                results.append({
                    "filename": f,
                    "content": content,
                    "timestamp": f.replace("summary_", "").replace(".md", ""),
                })
    return jsonify(results)


@app.route("/api/thesis/<run_id>")
def api_thesis(run_id):
    """Get a specific discovery thesis."""
    thesis_path = os.path.join(RESULTS_DIR, "discoveries", f"thesis_{run_id}.md")
    if os.path.exists(thesis_path):
        with open(thesis_path, "r", encoding="utf-8") as f:
            return jsonify({"content": f.read()})
    return jsonify({"error": "Thesis not found"}), 404


@app.route("/api/logs")
def api_logs():
    """Get current log history."""
    return jsonify(log_history)


@socketio.on("connect")
def handle_connect():
    """Send current state when a client connects."""
    emit("progress", current_progress)
    for entry in log_history[-50:]:  # Send last 50 logs
        emit("log", entry)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "discoveries"), exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    print("╔══════════════════════════════════════════════════╗")
    print("║   🔬 AI Science Discovery Team — Dashboard      ║")
    print("║   Open: http://localhost:5050                    ║")
    print("║   Make sure LM Studio is running on port 1234   ║")
    print("╚══════════════════════════════════════════════════╝")

    socketio.run(app, host="0.0.0.0", port=5050, debug=False)
