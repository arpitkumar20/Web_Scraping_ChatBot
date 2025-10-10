import logging
import threading
from flask import jsonify

from app.services.genai_response import handle_user_query
from app.services.vectordb_retrive import query_pinecone_index
from app.services.wati_api_service import send_whatsapp_message_v2
from app.services.select_namespace import run_namespace_selector

from app.models.postgresql_db import PostgreSQL
# from app.models.mysql_db import MySQL

# ✅ Call insert_message_data after successful processing
DB = PostgreSQL()
# DB = MySQL()
# Keep track of processed message IDs to prevent duplicate processing
processed_message_ids = set()

def handle_wati_webhook(data: dict) -> dict:
    try:
        message_id = data.get('id')
        phone_number = data.get('waId')
        message_text = data.get('text')

        # Immediate 200 OK response
        response = {"status": "received"}

        if not message_id or not phone_number or not message_text:
            return {"error": "Invalid payload structure"}

        if message_id in processed_message_ids:
            print(f"[INFO] Duplicate message detected: {message_id}. Skipping processing.")
            return response

        processed_message_ids.add(message_id)

        if len(processed_message_ids) > 10000:
            processed_message_ids.clear()

        def process_message():
            try:
                logging.info(f"Processing new message: {message_id}")

                # ------------------------------
                # Select namespace dynamically
                # ------------------------------
                namespace = run_namespace_selector(namespaces_file='web_info/web_info.json')
                logging.info(f"Selected namespace: {namespace}")

                query_response = query_pinecone_index(query_text=message_text, namespace=namespace)
                logging.info("Pinecone query completed.")

                genai_response = handle_user_query(retrieved_context=query_response, query=message_text)
                logging.info("GenAI response generated.")

                send_result = send_whatsapp_message_v2(phone_number=phone_number, message=genai_response)
                logging.info(f"WhatsApp message send result")

                filtered_message = {k: v for k, v in send_result['message'].items() if v is not None}

                # Add filtered 'message' dict into response
                response['result'] = send_result['result']
                response['message'] = filtered_message
                response['airesponse'] = genai_response
                logging.info("whatsapp message log is going to save into postgressql")
                data_to_insert = {
                    'status': response.get('status', 'received'),  # Or whatever default you want
                    'message': response['message'],
                    'airesponse': response['airesponse'],
                    'text': message_text
                }


                # insert_result = DB.insert_message_data(data_to_insert)
                insert_result = DB.insert_rds_message_data(data_to_insert)
                logging.info(f"Insert result: {insert_result}")

            except Exception as e:
                logging.error(f"Error processing message {message_id}: {str(e)}")

        threading.Thread(target=process_message).start()

        return response

    except Exception as e:
        logging.error(f"Webhook error: {str(e)}")
        return {"error": str(e)}
    
'''

{
    'status': 'received',
    'ok': True,
    'result': 'success',
    'message': {
        'whatsappMessageId': 'wamid.HBgMOTE4MjQwNjUxNTc0FQIAERgSODlGODYyMEZFQTY5NjkyRjc2AA==',
        'localMessageId': '9a538255-dfb8-46d1-aae7-22ae91c2f0c9',
        'text': "Is there something else I can help you with?  I'm happy to assist.",
        'type': 'text',
        'time': '1757528750',
        'status': 1,
        'statusString': 'SENT',
        'isOwner': True,
        'ticketId': '68c174cfd30bf0dfbc5ea53f',
        'assignedId': '689cd4696c529546aa769dcc',
        'sourceType': 0,
        'isDeleted': False,
        'translationStatus': 0,
        'id': '68c1c2aeca00333f30330665',
        'tenantId': '482313',
        'created': '2025-09-10T18:25:50.3790731Z',
        'conversationId': '68c174ade720d5e0ee9c60b6',
        'channelType': 0
    }
}


'''