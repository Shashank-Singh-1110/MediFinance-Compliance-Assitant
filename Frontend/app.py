from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from RAG_Engine import RagEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = Flask(__name__)
CORS(app)

engine = None


def get_engine():
    global engine
    if engine is None:
        logger.info("Initialising RAG Engine...")
        engine = RagEngine()
        logger.info("RAG Engine ready.")
    return engine


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": "ready" if engine else "not loaded",
        "model": "all-mpnet-base-v2",
        "chunks": 794,
    })


@app.route("/stats", methods=["GET"])
def stats():
    eng = get_engine()
    try:
        count = eng.retriever.collection.count()
    except Exception:
        count = 794
    return jsonify({
        "chunks": count,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "documents": 40,
        "regulators": ["IRDAI", "CAG", "CGHS", "RBI", "SEBI", "NHA"],
    })


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query_text = data["query"].strip()
    mode = data.get("mode", "llama3")
    category_filter = data.get("category_filter") or None
    conversation_id = data.get("conversation_id", "default")

    if mode not in ("llama3", "roberta", "distilroberta"):
        return jsonify({"error": f"Unknown mode '{mode}'"}), 400

    if not query_text:
        return jsonify({"error": "Empty query"}), 400

    try:
        eng = get_engine()
        result = eng.query(
            query_text,
            mode=mode,
            conversation_id=conversation_id,
            category_filter=category_filter,
        )
        return jsonify({
            "query": result.query,
            "answer": result.answer,
            "citations": [str(c) for c in result.citations],
            "mode": result.mode,
            "chunks_used": result.chunks_used,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
        })
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting MediFinance API on http://localhost:5000")
    logger.info("Make sure Ollama is running: ollama serve")
    get_engine()
    app.run(host="0.0.0.0", port=8000, debug=False)