# app.py — corrected and hardened
from flask import Flask, render_template, request, jsonify
from tripletgen.summarize_and_qa import summarize_and_qa, load_models
import threading
import os
from pathlib import Path
import csv

app = Flask(__name__, template_folder="templates", static_folder="static")

# Warm models in background for quicker first request (non-blocking)
def _warm():
    try:
        # load models (will attempt HF download if needed; uses HF_TOKEN env var)
        load_models(checkpoints_clean="checkpoints_clean", load_qa=False)
        print("[warmup] models loaded (or attempted)")
    except Exception as e:
        print("Warmup load failed:", e)

threading.Thread(target=_warm, daemon=True).start()


# --- Preset loader (CSV or defaults) ---
def load_preset_config(preset_name: str, data_dir: str = "data"):
    """
    Reads data/<preset_name>.csv if present and returns a dict of kwargs
    to pass to summarize_and_qa. Falls back to sensible defaults.
    Mapping:
      - score_threshold -> entail_threshold (float)
      - max_candidates -> K (int, capped)
      - max_new_tokens -> max_new_tokens (int, optional)
      - num_beams -> num_beams (int, optional)
    """
    defaults = {
        "entail_threshold": 0.8,
        "K": 6,
        "num_beams": 6,
        "max_new_tokens": 96,
    }
    csv_path = Path(data_dir) / f"{preset_name}.csv"
    if not csv_path.exists():
        return defaults

    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            # take first non-empty row
            row = next(reader, None)
            if not row:
                return defaults

            cfg = defaults.copy()
            if "score_threshold" in row and row["score_threshold"].strip():
                try:
                    cfg["entail_threshold"] = float(row["score_threshold"])
                except Exception:
                    pass
            if "max_candidates" in row and row["max_candidates"].strip():
                try:
                    # cap K to avoid huge generation requests
                    cfg["K"] = max(1, min(12, int(float(row["max_candidates"]))))
                except Exception:
                    pass
            if "num_beams" in row and row["num_beams"].strip():
                try:
                    cfg["num_beams"] = max(1, int(float(row["num_beams"])))
                except Exception:
                    pass
            if "max_new_tokens" in row and row["max_new_tokens"].strip():
                try:
                    cfg["max_new_tokens"] = max(8, int(float(row["max_new_tokens"])))
                except Exception:
                    pass
            # ensure num_beams >= K for stable decoding behavior
            cfg["num_beams"] = max(cfg["num_beams"], cfg["K"])
            return cfg
    except Exception as e:
        print(f"[load_preset_config] failed to read {csv_path}: {e}")
        return defaults


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    payload = request.json or {}
    article = (payload.get("article") or "").strip()
    question = payload.get("question")
    preset = payload.get("preset", "balanced")

    if not article:
        return jsonify({"error": "no article provided"}), 400

    # load preset config (safe defaults if csv missing or unreadable)
    cfg = load_preset_config(preset)

    try:
        # call summarize_and_qa with mapped args
        res = summarize_and_qa(
            article=article,
            question=question,
            K=cfg.get("K", 6),
            num_beams=cfg.get("num_beams", 6),
            max_new_tokens=cfg.get("max_new_tokens", 96),
            entail_threshold=cfg.get("entail_threshold", 0.8),
            checkpoints_clean="checkpoints_clean",
        )
        return jsonify(res)
    except Exception as e:
        # never leak a full traceback; return concise error message
        print("[api_summarize] error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
