import os
import tempfile
from flask import Blueprint, request, jsonify

from app.core.logger import logger
# from app.services.wati_service import send_whatsapp_message , get_whatsapp_messages
from app.services.wati_api_service import send_whatsapp_message_v2, get_whatsapp_messages_v2, send_whatsapp_image_v2
from app.services.vectordb_retrive import query_pinecone_index
from app.services.genai_response import handle_user_query

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Chat API is alive!"})

@chat_bp.route("/send-message", methods=["POST"])
def send_message():
    data = request.json or {}

    phone_number = data.get("phone_number")
    message = data.get("message")

    if not phone_number or not message:
        return jsonify({"error": "Missing required fields (phone_number, message)"}), 400

    # result = send_whatsapp_message(phone_number, message)
    result = send_whatsapp_message_v2(phone_number, message)
    return jsonify(result)

@chat_bp.route("/send-image", methods=["POST"])
def send_image():
    phone_number = request.form.get('phone_number')
    image_file = request.files.get('image')
    caption = request.form.get('caption', 'Image from Admin')

    if not phone_number or not image_file:
        return jsonify({"error": "phone_number and image are required"}), 400

    # Save uploaded image temporarily
    temp_image_path = f"/tmp/{image_file.filename}"
    image_file.save(temp_image_path)

    # Call your send function
    response = send_whatsapp_image_v2(phone_number, temp_image_path, caption)

    return jsonify(response)


@chat_bp.route("/receive-message", methods=["POST"])
def receive_message():
    """
    Receive incoming WhatsApp messages via WATI webhook.
    """
    try:
        data = request.json  # WATI sends incoming message here
        if not data:
            return jsonify({"error": "Empty payload"}), 400

        # Extract sender number
        sender_number = data.get("sender_number")

        if not sender_number:
            logger.error("Missing required fields: sender_number")
            return jsonify({"error": "Missing required fields (sender_number)"}), 400

        logger.info("Fetching WhatsApp messages for sender_number: %s", sender_number)
        # result = get_whatsapp_messages(sender_number)
        result = get_whatsapp_messages_v2(sender_number)

        if result.get("last_user_message"):
            last_user_message = result["last_user_message"].get("text", "")
            logger.info("Last user message retrieved: %s", last_user_message)
        else:
            last_user_message = ""
            logger.warning("No last user message found for sender_number: %s", sender_number)

        if last_user_message:
            query_response = query_pinecone_index(last_user_message)
            logger.info("Pinecone Querying complete with last user message")

            genai_response = handle_user_query(retrieved_context=query_response, query=last_user_message)
            logger.info("Generating GenAI response complete based on Pinecone query %s",genai_response)

            app_result = send_whatsapp_message_v2(
                phone_number=sender_number,
                message="testing api for wati server"
            )
            logger.info("WhatsApp message send result: %s", app_result)
        else:
            logger.warning("Skipping Pinecone query and GenAI response since last_user_message is empty")
        return jsonify(app_result)

    except Exception as err:
        return jsonify({"error": str(err)}), 500

