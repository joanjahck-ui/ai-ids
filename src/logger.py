import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import LOGS_DIR


def get_logger(name: str = "ai_ids", level: int = logging.INFO):
    """Create a logger that writes to console and rotating file in logs/"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = Path(LOGS_DIR) / "ai_ids.log"
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
