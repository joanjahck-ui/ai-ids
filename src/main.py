"""Minimal entrypoint to verify project scaffold."""
from .config import BASE_DIR, DATA_DIR
from .logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info(f"BASE_DIR={BASE_DIR}")
    logger.info(f"DATA_DIR={DATA_DIR}")
    logger.info("AI-IDS scaffold ready. Proceed to data ingestion and preprocessing.")


if __name__ == "__main__":
    main()
