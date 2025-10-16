# import logging
# import threading
# from flask import jsonify

# from app.services.genai_response import handle_user_query
# from app.services.vectordb_retrive import query_pinecone_index
# from app.services.wati_api_service import send_whatsapp_message_v2
# from app.services.select_namespace import run_namespace_selector

# from app.models.postgresql_db import PostgreSQL
# # from app.models.mysql_db import MySQL

# # ✅ Call insert_message_data after successful processing
# DB = PostgreSQL()
# # DB = MySQL()
# # Keep track of processed message IDs to prevent duplicate processing
# processed_message_ids = set()

# def handle_wati_webhook(data: dict) -> dict:
#     try:
#         message_id = data.get('id')
#         phone_number = data.get('waId')
#         message_text = data.get('text')

#         # Immediate 200 OK response
#         response = {"status": "received"}

#         if not message_id or not phone_number or not message_text:
#             return {"error": "Invalid payload structure"}

#         if message_id in processed_message_ids:
#             print(f"[INFO] Duplicate message detected: {message_id}. Skipping processing.")
#             return response

#         processed_message_ids.add(message_id)

#         if len(processed_message_ids) > 10000:
#             processed_message_ids.clear()

#         def process_message():
#             try:
#                 logging.info(f"Processing new message: {message_id}")

#                 # ------------------------------
#                 # Select namespace dynamically
#                 # ------------------------------
#                 namespace = run_namespace_selector(namespaces_file='web_info/web_info.json')
#                 logging.info(f"Selected namespace: {namespace}")

#                 query_response = query_pinecone_index(query_text=message_text, namespace=namespace)
#                 logging.info("Pinecone query completed.")

#                 genai_response = handle_user_query(user_id=str(message_id), retrieved_context=query_response, query=message_text)
#                 logging.info("GenAI response generated.")

#                 send_result = send_whatsapp_message_v2(phone_number=phone_number, message=genai_response)
#                 logging.info(f"WhatsApp message send result")

#                 filtered_message = {k: v for k, v in send_result['message'].items() if v is not None}

#                 # Add filtered 'message' dict into response
#                 response['result'] = send_result['result']
#                 response['message'] = filtered_message
#                 response['airesponse'] = genai_response
#                 logging.info("whatsapp message log is going to save into postgressql")
#                 data_to_insert = {
#                     'status': response.get('status', 'received'),  # Or whatever default you want
#                     'message': response['message'],
#                     'airesponse': response['airesponse'],
#                     'text': message_text
#                 }


#                 # insert_result = DB.insert_message_data(data_to_insert)
#                 insert_result = DB.insert_rds_message_data(data_to_insert)
#                 logging.info(f"Insert result: {insert_result}")

#             except Exception as e:
#                 logging.error(f"Error processing message {message_id}: {str(e)}")

#         threading.Thread(target=process_message).start()

#         return response

#     except Exception as e:
#         logging.error(f"Webhook error: {str(e)}")
#         return {"error": str(e)}
    
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


# import logging
# import threading
# from flask import jsonify

# from app.services.genai_response import handle_user_query
# from app.services.vectordb_retrive import query_pinecone_index
# from app.services.wati_api_service import send_whatsapp_message_v2
# from app.services.select_namespace import run_namespace_selector

# from app.models.postgresql_db import PostgreSQL
 
# from app.services.booking_service import validate_and_book
# from app.services.genai_response import detect_booking_intent_and_fields
# from app.services.booking_service import validate_and_book
 
 
# # from app.models.mysql_db import MySQL

# # ✅ Call insert_message_data after successful processing
# DB = PostgreSQL()
# # DB = MySQL()
# # Keep track of processed message IDs to prevent duplicate processing
# processed_message_ids = set()

# def handle_wati_webhook(data: dict) -> dict:
#     try:
#         message_id = data.get('id')
#         phone_number = data.get('waId')
#         message_text = data.get('text')

#         # Immediate 200 OK response
#         response = {"status": "received"}

#         if not message_id or not phone_number or not message_text:
#             return {"error": "Invalid payload structure"}

#         if message_id in processed_message_ids:
#             print(f"[INFO] Duplicate message detected: {message_id}. Skipping processing.")
#             return response

#         processed_message_ids.add(message_id)

#         if len(processed_message_ids) > 10000:
#             processed_message_ids.clear()

#         def process_message():
#             try:
#                 logging.info(f"Processing new message: {message_id}")

#                 # ------------------------------
#                 # Select namespace dynamically
#                 # ------------------------------
#                 namespace = run_namespace_selector(namespaces_file='web_info/web_info.json')
#                 logging.info(f"Selected namespace: {namespace}")



#                 # 1) Ask LLM once to decide booking vs other AND extract fields (if booking)
#                 # First, get a small context for the extractor (reuse what you already do)
#                 context_for_extractor = query_pinecone_index(query_text=message_text, namespace=namespace)

#                 intent_payload = detect_booking_intent_and_fields(user_id=str(message_id),
#                     retrieved_context=context_for_extractor,
#                     query=message_text
#                 )

#                 if intent_payload.get("intent") == "book":
#                     doctor       = intent_payload.get("doctor")
#                     date_str     = intent_payload.get("date")
#                     start_hhmm   = intent_payload.get("start")
#                     duration_min = intent_payload.get("duration_min") or 30
#                     win_start    = intent_payload.get("window_start")
#                     win_end      = intent_payload.get("window_end")

#                     # Missing fields? Ask the user neatly and stop here.
#                     missing = []
#                     if not doctor:     missing.append("doctor")
#                     if not date_str:   missing.append("date (YYYY-MM-DD)")
#                     if not start_hhmm: missing.append("start (HH:MM 24h)")
#                     if not win_start:  missing.append("window_start (HH:MM 24h)")
#                     if not win_end:    missing.append("window_end (HH:MM 24h)")

#                     if missing:
#                         prompt = (
#                             "I need a bit more info to book your appointment.\n"
#                             f"Missing: {', '.join(missing)}.\n"
#                             "Please reply with: doctor name, date (YYYY-MM-DD), start time (HH:MM 24h)."
#                         )
#                         send_result = send_whatsapp_message_v2(phone_number=phone_number, message=prompt)
#                         filtered_message = {k: v for k, v in send_result.get('message', {}).items() if v is not None}

#                         response['result'] = send_result.get('result')
#                         response['message'] = filtered_message
#                         response['airesponse'] = prompt

#                         data_to_insert = {
#                             'status': response.get('status', 'received'),
#                             'message': response['message'],
#                             'airesponse': response['airesponse'],
#                             'text': message_text
#                         }
#                         insert_result = DB.insert_rds_message_data(data_to_insert)
#                         logging.info(f"Insert result (booking-missing): {insert_result}")
#                         return  # short-circuit booking branch

#                     # We have enough → deterministic booking call
#                     result = validate_and_book(
#                         namespace=namespace,
#                         doctor=doctor,
#                         date=date_str,
#                         start=start_hhmm,
#                         duration_min=int(duration_min),
#                         window_start=win_start,
#                         window_end=win_end,
#                         booked_by=phone_number
#                     )

#                     send_result = send_whatsapp_message_v2(phone_number=phone_number, message=result["message"])
#                     filtered_message = {k: v for k, v in send_result.get('message', {}).items() if v is not None}

#                     response['result'] = send_result.get('result')
#                     response['message'] = filtered_message
#                     response['airesponse'] = result["message"]

#                     data_to_insert = {
#                         'status': response.get('status', 'received'),
#                         'message': response['message'],
#                         'airesponse': response['airesponse'],
#                         'text': message_text
#                     }
#                     insert_result = DB.insert_rds_message_data(data_to_insert)
#                     logging.info(f"Insert result (booking): {insert_result}")

#                     return  # short-circuit; do NOT run normal RA



#                 query_response = query_pinecone_index(query_text=message_text, namespace=namespace)
#                 logging.info("Pinecone query completed.")

#                 genai_response = handle_user_query(str(message_id), retrieved_context=query_response, query=message_text)
#                 logging.info("GenAI response generated.")

#                 send_result = send_whatsapp_message_v2(phone_number=phone_number, message=genai_response)
#                 logging.info(f"WhatsApp message send result")

#                 filtered_message = {k: v for k, v in send_result['message'].items() if v is not None}

#                 # Add filtered 'message' dict into response
#                 response['result'] = send_result['result']
#                 response['message'] = filtered_message
#                 response['airesponse'] = genai_response
#                 logging.info("whatsapp message log is going to save into postgressql")
#                 data_to_insert = {
#                     'status': response.get('status', 'received'),  # Or whatever default you want
#                     'message': response['message'],
#                     'airesponse': response['airesponse'],
#                     'text': message_text
#                 }


#                 # insert_result = DB.insert_message_data(data_to_insert)
#                 insert_result = DB.insert_rds_message_data(data_to_insert)
#                 logging.info(f"Insert result: {insert_result}")

#             except Exception as e:
#                 logging.error(f"Error processing message {message_id}: {str(e)}")

#         threading.Thread(target=process_message).start()

#         return response

#     except Exception as e:
#         logging.error(f"Webhook error: {str(e)}")
#         return {"error": str(e)}



import logging
import threading
from typing import Dict

from app.services.genai_response import (
    handle_user_query
)
from app.services.vectordb_retrive import query_pinecone_index
from app.services.wati_api_service import send_whatsapp_message_v2
from app.services.select_namespace import run_namespace_selector
from app.models.postgresql_db import PostgreSQL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB = PostgreSQL()
processed_messages = set()
MAX_PROCESSED_CACHE = 10000

def handle_wati_webhook(data: dict) -> dict:
    """
    Optimized webhook handler with memory-aware processing.
    
    Flow:
    1. Validate and deduplicate
    2. Detect booking intent with memory context
    3. Handle booking OR general query
    4. Save to database
    """
    try:
        message_id = data.get('id')
        phone = data.get('waId')
        text = data.get('text')
        
        # Immediate acknowledgment
        response = {"status": "received"}
        
        # Validation
        if not all([message_id, phone, text]):
            return {"error": "Invalid payload"}
        
        # Deduplication
        if message_id in processed_messages:
            logger.info(f"Duplicate: {message_id}")
            return response
        
        processed_messages.add(message_id)
        
        # Cache cleanup
        if len(processed_messages) > MAX_PROCESSED_CACHE:
            processed_messages.clear()
        
        # Async processing
        threading.Thread(target=_process_message, args=(message_id, phone, text, response)).start()
        
        return response
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"error": str(e)}

def _process_message(msg_id: str, phone: str, text: str, response: dict):
    """Process message with memory context"""
    try:
        logger.info(f"Processing: {msg_id}")
        
        # Select namespace
        namespace = run_namespace_selector(namespaces_file='web_info/web_info.json')
        logger.info(f"Namespace: {namespace}")
        
        # Get context for intent detection
        context = query_pinecone_index(query_text=text, namespace=namespace)
       
        # Handle user query with context
        ai_response = handle_user_query(
            user_id=phone,
            retrieved_context=context,
            query=text
        )
        
        # Send response
        send_result = send_whatsapp_message_v2(phone, ai_response)
        logger.info("WhatsApp sent")
        
        # Prepare response data
        filtered_msg = {
            k: v for k, v in send_result.get('message', {}).items() 
            if v is not None
        }
        
        response['result'] = send_result.get('result')
        response['message'] = filtered_msg
        response['airesponse'] = ai_response
        
        # Save to database
        db_data = {
            'status': response.get('status', 'received'),
            'message': response['message'],
            'airesponse': response['airesponse'],
            'text': text
        }
        
        DB.insert_rds_message_data(db_data)
        logger.info("Saved to DB")
        
    except Exception as e:
        logger.error(f"Processing error {msg_id}: {e}")