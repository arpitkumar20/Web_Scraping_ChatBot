import logging
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify
from app.services.wati_webhook import handle_wati_webhook

wati_bp = Blueprint("wati", __name__)

load_dotenv()

@wati_bp.route("/wati_webhook", methods=["POST"])
def wati_webhook():
    try:
        data = request.json
        print('Incoming webhook data:', data)

        response = handle_wati_webhook(data)
        return jsonify(response), 200
    except Exception as e:
        logging.error(f"Unhandled webhook exception: {str(e)}")
        return jsonify({"error": str(e)}), 500
