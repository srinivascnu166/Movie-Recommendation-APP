import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a console handler
    # Check if console handler already exists to avoid duplicate logging
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create a formatter for the console output
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(ch)

    return logger