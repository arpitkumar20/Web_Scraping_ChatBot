from flask import Blueprint, request, jsonify

from app.core.logger import logger
from app.services.wati_webhook import handle_wati_webhook

wati_bp = Blueprint("wati", __name__)

@wati_bp.route("/wati_webhook", methods=["POST"])
def wati_webhook():
    try:
        data = request.json
        print('Incoming webhook data:', data)

        response = handle_wati_webhook(data)
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Unhandled webhook exception: {str(e)}")
        return jsonify({"error": str(e)}), 500
