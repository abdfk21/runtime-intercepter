"""Structured JSON logging and observability."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON without exposing credentials."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include structured extra fields if present
        extra_keys = [
            "trace_id",
            "span_id",
            "handoff_id",
            "source_agent",
            "target_agent",
            "model",
            "architecture",
            "input_hash",
            "output_hash",
            "signal_scores",
            "decision",
            "latency_ms",
            "repair_attempts",
            "fault_type",
            "condition",
        ]
        for key in extra_keys:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str = "runtime_interceptor") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
