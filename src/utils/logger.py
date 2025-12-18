"""
Structured logging setup.
"""

import logging
import logging.config
from pathlib import Path


def setup_logging(level: str = "INFO"):
    """
    Configure logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    Path("logs").mkdir(exist_ok=True)
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": "logs/app.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)
