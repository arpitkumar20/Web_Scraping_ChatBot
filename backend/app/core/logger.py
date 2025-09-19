import logging
import os
from logging.handlers import RotatingFileHandler

# ------------------ Logging Setup ------------------
LOG_FOLDER = "logs"
LOG_FILE = "app.log"

# Create logs folder if not exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Full path for log file
log_path = os.path.join(LOG_FOLDER, LOG_FILE)

# Create custom logger
logger = logging.getLogger("my_project_logger")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if re-imported
if not logger.handlers:

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler (rotating logs, max 5MB each, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    # Common format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Export logger
__all__ = ["logger"]
