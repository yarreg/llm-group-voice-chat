import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional
from functools import wraps
import asyncio

from .config import get_config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request_id if available
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id

        # Add extra fields
        if hasattr(record, 'duration_ms'):
            log_entry["duration_ms"] = record.duration_ms

        if hasattr(record, 'audio_duration_ms'):
            log_entry["audio_duration_ms"] = record.audio_duration_ms

        if hasattr(record, 'voice_id'):
            log_entry["voice_id"] = record.voice_id

        if hasattr(record, 'text_length'):
            log_entry["text_length"] = record.text_length

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'getMessage',
                          'exc_info', 'exc_text', 'stack_info', 'message',
                          'asctime', 'request_id', 'duration_ms', 'audio_duration_ms',
                          'voice_id', 'text_length']:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging():
    """Setup logging with JSON formatter"""
    config = get_config()

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.logging.level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # Setup uvicorn access logger
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name"""
    return logging.getLogger(name)


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, **extra_fields):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000
        logger.info(
            f"Operation completed: {operation}",
            extra={
                "duration_ms": duration,
                "operation": operation,
                **extra_fields
            }
        )


def log_timing(logger: logging.Logger, operation: str):
    """Decorator for timing function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.info(
                    f"Function completed: {operation}",
                    extra={
                        "duration_ms": duration,
                        "operation": operation,
                        "function": func.__name__
                    }
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Function failed: {operation}",
                    extra={
                        "duration_ms": duration,
                        "operation": operation,
                        "function": func.__name__,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise
        return wrapper

    return decorator


async def log_timing_async(logger: logging.Logger, operation: str):
    """Decorator for timing async function execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.info(
                    f"Async function completed: {operation}",
                    extra={
                        "duration_ms": duration,
                        "operation": operation,
                        "function": func.__name__
                    }
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Async function failed: {operation}",
                    extra={
                        "duration_ms": duration,
                        "operation": operation,
                        "function": func.__name__,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise
        return wrapper

    return decorator


class RequestLogger:
    """Helper class for request-scoped logging"""

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.logger = get_logger("tts.request")

    def info(self, message: str, **extra_fields):
        """Log info message with request_id"""
        self.logger.info(message, extra={"request_id": self.request_id, **extra_fields})

    def error(self, message: str, **extra_fields):
        """Log error message with request_id"""
        self.logger.error(message, extra={"request_id": self.request_id, **extra_fields})

    def warning(self, message: str, **extra_fields):
        """Log warning message with request_id"""
        self.logger.warning(message, extra={"request_id": self.request_id, **extra_fields})

    def debug(self, message: str, **extra_fields):
        """Log debug message with request_id"""
        self.logger.debug(message, extra={"request_id": self.request_id, **extra_fields})