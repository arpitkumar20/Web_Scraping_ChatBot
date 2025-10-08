import os
import requests
import logging as log

from dotenv import load_dotenv

load_dotenv()

TIKA_URL = os.getenv('TIKA_URL')

if not all([TIKA_URL]):
    log.error('ENV TIKA_URL missing')
    raise EnvironmentError('Missing required environment variable')

# Create a session for connection pooling
session = requests.Session()

TIMEOUT = 300
# Tika client function
def extract_text_with_tika_client(file_path):
    try:
        with open(file_path, 'rb') as file:
            headers = {'Accept': 'text/plain'}
            response = session.put(url=TIKA_URL, data=file, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
    except requests.RequestException as e:
        log.error(f'Error extracting text with Tika: {e}')
        raise Exception(f'error extracting text from file: {file_path} with tika server.')