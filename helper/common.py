import datetime
import logging
import os
import socket
from typing import (
    Tuple
)

# Set environment variables for S3 support
os.environ["S3_BUCKET_NAME"] = "psr-factory-helper-backup"

_DEBUG_LOG = True

ASSETS_FOLDER = "assets"
LOGS_PATH = 'logs'
CACHE_FOLDER = "cache"
CONTAINER_PERSISTENT_VOLUME_FOLDER = "/factory-helper-volume"
TEST_PERSISTENT_VOLUME_FOLDER = os.path.join(CACHE_FOLDER, "volume")
VECTORSTORES_FOLDER = "vectorstores"


def get_machine_ip() -> Tuple[str, str]:
    """
    Get the machine's IP address and hostname.
    Returns a tuple containing the IP address and hostname.
    """
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
        hostname = socket.gethostname()
        return ip_address, hostname
    except Exception as e:
        logging.error(f"Error getting machine IP: {e}")
        return "unknown", "unknown"


def get_persistent_cache_path() -> str:
    if os.path.exists(CONTAINER_PERSISTENT_VOLUME_FOLDER):
        return CONTAINER_PERSISTENT_VOLUME_FOLDER

    os.makedirs(TEST_PERSISTENT_VOLUME_FOLDER, exist_ok=True)
    return TEST_PERSISTENT_VOLUME_FOLDER

def get_vectorstore_base_path() -> str:
    """
    Get the base path for vectorstores. In containers, stores in persistent volume.
    Otherwise stores in local vectorstores folder.
    """
    if os.path.exists(CONTAINER_PERSISTENT_VOLUME_FOLDER):
        # Container mode: store in persistent volume
        vectorstore_path = os.path.join(CONTAINER_PERSISTENT_VOLUME_FOLDER, VECTORSTORES_FOLDER)
        os.makedirs(vectorstore_path, exist_ok=True)
        return vectorstore_path

    # Local/development mode: store in local vectorstores folder
    os.makedirs(VECTORSTORES_FOLDER, exist_ok=True)
    return VECTORSTORES_FOLDER

def debug_mode() -> bool:
    """
    Check if the application is running in debug mode.
    Returns True if running in production, False otherwise.
    """
    return os.getenv("HELPER_DEBUG", "false").lower() == "true"


def get_logger(context: str = "general"):
    context = context if len(context) > 0 else "general"
    logger = logging.getLogger(context)
    if not logger.hasHandlers():
        logs_path = os.path.join(get_persistent_cache_path(), LOGS_PATH)
        # Create the logs folder if it doesn't exist
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # Generate the log filename with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"log_{context}_{timestamp}.txt"
        log_path = os.path.join(logs_path, log_filename)

        # Configure the logging module to write to the log file
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        logger.setLevel(logging.DEBUG)

        if _DEBUG_LOG:
            # Create a stream handler to log to standard output
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # Create a file handler to log to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger