from dotenv import load_dotenv
from flask import Blueprint, request, jsonify
import logging

from app.core.logging import get_logger
from app.services.wati_webhook import handle_wati_webhook
from app.services.genai_response import (
    clear_user_memory,
    get_user_history
)

load_dotenv()

wati_bp = Blueprint("wati", __name__)
logger = get_logger(__name__)

@wati_bp.route("/wati_webhook", methods=["POST"])
def wati_webhook():
    """
    Main webhook endpoint for WATI messages.
    Handles incoming WhatsApp messages with memory-aware processing.
    """
    try:
        data = request.json
        logger.info(f"Webhook received: {data.get('id')}")
        
        response = handle_wati_webhook(data)
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Webhook exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

@wati_bp.route("/clear_history/<user_id>", methods=["DELETE"])
def clear_history(user_id: str):
    """
    Clear conversation history for a user.
    Useful for testing or user-requested resets.
    """
    try:
        clear_user_memory(user_id)
        return jsonify({
            "status": "success",
            "message": f"History cleared for {user_id}"
        }), 200
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({"error": str(e)}), 500

@wati_bp.route("/history/<user_id>", methods=["GET"])
def get_history(user_id: str):
    """
    Get conversation history for a user.
    Returns recent exchanges and summary.
    """
    try:
        history = get_user_history(user_id)
        return jsonify({
            "user_id": user_id,
            "conversations": history,
            "count": len(history)
        }), 200
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": str(e)}), 500

@wati_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200