from flask import Blueprint, jsonify

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Admin API is healthy!"})

