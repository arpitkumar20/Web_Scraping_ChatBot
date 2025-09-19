import os
import json
import bson
import decimal
import datetime
import logging
from typing import Dict, List, Union

# ------------------ Configure Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class COMMON:
    def stringify(item):
        if isinstance(item, (bson.objectid.ObjectId,datetime.datetime,datetime.date,datetime.time,datetime.timezone,decimal.Decimal)):
            return str(item)
        
        return item

    def save_json_data(new_data: Union[Dict, List[Dict]]) -> None:
        """
        Save or update JSON data into 'data/storage.json'.
        Automatically creates the folder if it doesn't exist,
        prevents duplicates (checks by 'url' or 'namespace'),
        and logs every action.

        Args:
            new_data (dict | list[dict]): New data (single dict or list of dicts).
        """

        folder_path = "web_info"
        filename = "web_info.json"

        # Ensure data is always a list
        if isinstance(new_data, dict):
            new_data = [new_data]

        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Full file path
        file_path = os.path.join(folder_path, filename)

        # Load old data if file exists
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    old_data = json.load(file)
                    if not isinstance(old_data, list):
                        old_data = [old_data]
                    logging.info(f"Loaded {len(old_data)} existing records from {file_path}")
                except json.JSONDecodeError:
                    logging.warning(f"File {file_path} is empty or corrupted. Starting fresh.")
                    old_data = []
        else:
            logging.info(f"No existing file found. Creating new one at {file_path}")
            old_data = []

        # Merge logic: check duplicates by "url" or "namespace"
        def get_unique_key(item: Dict) -> str:
            return item.get("url") or item.get("namespace") or ""

        existing_map = {get_unique_key(item): item for item in old_data if get_unique_key(item)}

        for item in new_data:
            key = get_unique_key(item)
            if key and key in existing_map:
                # Update existing record (merge dicts)
                existing_map[key].update(item)
                logging.info(f"Updated record with key: {key}")
            else:
                # Add new record
                old_data.append(item)
                if key:
                    existing_map[key] = item
                    logging.info(f"Added new record with key: {key}")
                else:
                    logging.info("Added new record without unique key")

        # Save updated data
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(old_data, file, indent=4, ensure_ascii=False)

        logging.info(f"Saved total {len(old_data)} records into {file_path}")

