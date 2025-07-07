import logging
import os
from datetime import datetime

# Create log directory (within app container – ephemeral on Render)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create unique log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Formatter for logs
formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

# File Handler (logs saved while app is running)
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(formatter)

# Console Handler (for Render Logs tab)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Root logger setup
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Optional: expose logger instance
logger = logging.getLogger(__name__)
