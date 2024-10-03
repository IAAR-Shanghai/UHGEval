import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "eval.log",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "eval": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("eval")
