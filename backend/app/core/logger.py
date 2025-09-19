import logging
import os
import time
from logging.handlers import RotatingFileHandler

# ------------------ Logging Setup ------------------
LOG_FOLDER = "logs"
LOG_FILE = "app.log"
LOG_RETENTION_DAYS = 2  # keep logs only for 2 days

# Create logs folder if not exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Full path for log file
log_path = os.path.join(LOG_FOLDER, LOG_FILE)


def cleanup_old_logs(folder: str, retention_days: int = 2):
    """Delete log files older than retention_days."""
    now = time.time()
    cutoff = now - (retention_days * 86400)  # days → seconds

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.startswith("app.log"):
            file_mtime = os.path.getmtime(file_path)
            if file_mtime < cutoff:
                try:
                    os.remove(file_path)
                    print(f"[Log Cleanup] Deleted old log file: {file_path}")
                except Exception as e:
                    print(f"[Log Cleanup] Failed to delete {file_path}: {e}")


# Run cleanup before logger setup
cleanup_old_logs(LOG_FOLDER, LOG_RETENTION_DAYS)

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
