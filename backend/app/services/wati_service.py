import requests
import os
# from urllib.parse import quote
from urllib.parse import unquote


API_KEY = os.getenv('WATI_APY_KEY')
BASE_URL = "https://app-server.wati.io"
    
def send_whatsapp_message(phone_number, message) -> dict:
    """
    Send a WhatsApp session message using the WATI API.
    """
    if not isinstance(message, str):
        # Convert to string if a dict or other type is passed
        message = str(message)
    # URL encode the message
    encoded_message = unquote(message)

    # url = f"{BASE_URL}/sendSessionMessage/{phone_number}?messageText={encoded_message}"
    url = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded_message}"


    headers = {
        "accept": "*/*",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
def get_whatsapp_messages(phone_number) -> dict:
    """
    Fetch WhatsApp messages from WATI API for a specific phone number.
    """
    url = f"{BASE_URL}/api/v1/getMessages/{phone_number}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                # Filter only user messages
                all_items = data.get("messages", {}).get("items", [])

                user_messages = [
                    {
                        "conversationId": item.get("conversationId"),
                        "id": item.get("id"),
                        "owner": item.get("owner"),
                        "status": item.get("statusString"),
                        "text": item.get("text"),
                        "ticketId": item.get("ticketId"),
                        "timestamp": item.get("timestamp")
                    }
                    for item in all_items
                    if item.get("eventType") == "message" and item.get("type") == "text" and item.get("owner") is False
                ]

                if user_messages:
                    # Get the latest user message by timestamp
                    last_message = max(user_messages, key=lambda x: int(x['timestamp']))
                    return {"last_user_message": last_message}

                return {"last_user_message": None}
            # return response.json()   # Parsed JSON response
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}






# curl -X 'POST' \
#   'https://app-server.wati.io/api/v1/sendSessionMessage/919669092627?messageText=Hi%20testing' \
#   -H 'accept: */*' \
#   -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLlE6Vc' \
#   -d ''