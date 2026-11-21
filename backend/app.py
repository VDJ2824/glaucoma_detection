"""Minimal development API. Gemini integration is intentionally unimplemented."""
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS


def create_app():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")}})

    @app.get("/api/health")
    def health():
        return jsonify(status="scaffold", reports_enabled=False)

    @app.post("/api/reports")
    def report():
        # Fail closed until consent and strict payload validation are implemented.
        return jsonify(error="Report generation is not implemented."), 501

    return app


if __name__ == "__main__":
    create_app().run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("FLASK_PORT", "5000")), debug=False)
