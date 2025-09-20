# app.py
import os
import traceback
import logging
from flask import Flask, request, jsonify, render_template
from tripletgen.summarize_and_qa import summarize_and_qa, load_models, MODELS
import threading

# Optional: force dev generator quickly by uncommenting below or set env var outside.
# os.environ["GENERATOR_MODEL"] = "sshleifer/tiny-bart"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    mc = MODELS
    return jsonify({"ok": True, "models_ready": getattr(mc, "ready", False), "models_loading": getattr(mc, "loading", False), "last_error": getattr(mc, "last_error", None)})


@app.route("/debug-load", methods=["GET", "POST"])
def debug_load():
    """
    Synchronously load models and return JSON with status or traceback.
    Use this to get the real error output when a model stalls.
    """
    try:
        checkpoints_clean = request.args.get("checkpoints_clean", "checkpoints_clean")
        mc = load_models(checkpoints_clean=checkpoints_clean, load_qa=True)
        return jsonify({"status": "ok" if getattr(mc, "ready", False) else "partial", "models_ready": getattr(mc, "ready", False), "models_loading": getattr(mc, "loading", False), "last_error": getattr(mc, "last_error", None)})
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("debug-load failed")
        return jsonify({"status": "error", "exception": str(e), "traceback": tb}), 500


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    try:
        payload = request.get_json(force=True)
        article = payload.get("article", "")
        question = payload.get("question", None)
        if not article:
            return jsonify({"error": "Missing article"}), 400

        max_new_tokens = min(512, max(16, int(payload.get("max_new_tokens", 512))))
        num_beams = min(12, max(1, int(payload.get("num_beams", 6))))
        length_penalty = min(3.0, max(0.2, float(payload.get("length_penalty", 1.2))))
        do_sample = bool(payload.get("do_sample", False))
        top_p = float(payload.get("top_p", 0.95))
        temperature = float(payload.get("temperature", 0.7))
        K = min(8, max(1, int(payload.get("K", 3))))

        res = summarize_and_qa(article=article, question=question, K=K, num_beams=num_beams, max_new_tokens=max_new_tokens,
                               entail_threshold=0.8, checkpoints_clean="checkpoints_clean", do_sample=do_sample,
                               top_p=top_p, temperature=temperature, length_penalty=length_penalty)
        return jsonify(res)
    except Exception as e:
        logger.exception("api_summarize error")
        return jsonify({"error": "internal server error", "detail": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    # Warm models in background thread for normal startup (keeps server responsive).
    def _warm():
        try:
            load_models(checkpoints_clean="checkpoints_clean", load_qa=True)
            logger.info("Background warmup complete.")
        except Exception:
            logger.exception("Background warmup failed.")

    t = threading.Thread(target=_warm, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8080, threaded=True)
