import logging
import os
import queue
from datetime import datetime


# ── SSE Queue: holds live log messages for frontend streaming ──
log_queue = queue.Queue()


# ── App-level logger names (only these get pushed to SSE) ──
APP_LOGGERS = {
    'TrainModel', 'ModelTuner', 'FileOperation',
    'PredictModel', 'LoadValidate', 'Preprocessor',
    'HFUploader',
}

# ── Custom handler that pushes app log records into the SSE queue ──
class _QueueHandler(logging.Handler):
    def emit(self, record):
        if record.name in APP_LOGGERS:
            log_queue.put(self.format(record))


# ── Create logs directory ──
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Timestamped log file ──
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
log_file_path = os.path.join(LOG_DIR, f"log_{CURRENT_TIME_STAMP}.log")

# ── Configure root logger (file + console + SSE queue) ──
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler(),
        _QueueHandler(),
    ],
    force=True
)
