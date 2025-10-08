# import tempfile
# import os
# import time
# from app.helper.utils import COMMON
# from app.core.logging import get_logger
# from app.services.tika_rest_client import extract_text_with_tika_client

# logger = get_logger(__name__)

# # --------------------- Helper Function ---------------------
# def process_file(file) -> dict:
#     """
#     Process uploaded file and return dictionary with file info and extracted text.
#     """
#     run_hash = COMMON.get_hash()
#     filename = file.filename

#     with tempfile.TemporaryDirectory(prefix=run_hash) as folder_path:
#         file_path = os.path.join(folder_path, filename)
#         file.save(file_path)
#         file_size = os.path.getsize(file_path)

#         # --- Extract text ---
#         start_time = time.time()
#         try:
#             if filename.lower().endswith(('.txt', '.csv')):
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                     text = f.read()
#                 logger.info(f'[TXT/CSV] Extracted in {time.time() - start_time:.4f}s')
#             else:
#                 text = extract_text_with_tika_client(file_path)
#                 logger.info(f'[TIKA] Extracted in {time.time() - start_time:.4f}s')
#         except Exception as e:
#             logger.error(f'Error extracting text from {filename}: {e}')
#             raise RuntimeError(f'Text extraction failed: {str(e)}')

#         # --- Clean text ---
#         cleaned_text = ' '.join(line.strip() for line in str(text).splitlines() if line.strip())

#         # --- Prepare result dictionary ---
#         result = {
#             'run_hash': run_hash,
#             'filename': filename,
#             'file_path': file_path,
#             'file_size': file_size,
#             'extracted_text': cleaned_text
#         }

#         logger.info(f'File processed successfully: {filename}, hash: {run_hash}')
#         return result



import pandas as pd
import tempfile
import os
import time

from app.helper.utils import COMMON
from app.core.logging import get_logger
from app.services.tika_rest_client import extract_text_with_tika_client

logger = get_logger(__name__)

def process_file(file) -> dict:
    """
    Process uploaded file and return dictionary with file info and extracted text.
    For XLSX/CSV files, preserves row-column structure in a clean string format.
    """
    run_hash = COMMON.get_hash()
    filename = file.filename

    with tempfile.TemporaryDirectory(prefix=run_hash) as folder_path:
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        file_size = os.path.getsize(file_path)

        # --- Extract text ---
        start_time = time.time()
        try:
            if filename.lower().endswith('.txt'):
                # Simple text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                logger.info(f'[TXT] Extracted in {time.time() - start_time:.4f}s')

            elif filename.lower().endswith('.csv'):
                # CSV file: preserve rows and columns
                df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
                # Clean whitespace in each cell
                df = df.applymap(lambda x: str(x).strip())
                # Convert to tabular string
                text = df.to_csv(index=False)
                logger.info(f'[CSV] Extracted in {time.time() - start_time:.4f}s')

            elif filename.lower().endswith(('.xlsx', '.xls')):
                # Excel file
                df = pd.read_excel(file_path, dtype=str)
                df = df.applymap(lambda x: str(x).strip())
                text = df.to_csv(index=False)
                logger.info(f'[XLSX] Extracted in {time.time() - start_time:.4f}s')

            else:
                # Other file types use Tika
                text = extract_text_with_tika_client(file_path)
                logger.info(f'[TIKA] Extracted in {time.time() - start_time:.4f}s')

        except Exception as e:
            logger.error(f'Error extracting text from {filename}: {e}')
            raise RuntimeError(f'Text extraction failed: {str(e)}')

        # --- Clean text ---
        cleaned_text = ' '.join(line.strip() for line in str(text).splitlines() if line.strip())

        # --- Prepare result dictionary ---
        result = {
            'run_hash': run_hash,
            'filename': filename,
            'file_path': file_path,
            'file_size': file_size,
            'extracted_text': cleaned_text
        }

        logger.info(f'File processed successfully: {filename}, hash: {run_hash}')
        return result
