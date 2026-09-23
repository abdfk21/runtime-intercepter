"""Security utilities, input sanitation, and secret masking."""

import hashlib
import re
from typing import Any, Dict


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Mask sensitive keys for safe display or logging."""
    if not secret:
        return ""
    if len(secret) <= visible_chars * 2:
        return "***"
    return f"{secret[:visible_chars]}...{secret[-visible_chars:]}"


def sanitize_text_input(text: str) -> str:
    """Strip dangerous characters and control sequences from user or external inputs."""
    if not isinstance(text, str):
        return str(text)
    # Remove null bytes and excessive control characters
    text = text.replace("\x00", "")
    return text.strip()


def compute_content_hash(content: Any) -> str:
    """Generate deterministic SHA-256 hash of text or serialized structure."""
    if isinstance(content, (dict, list)):
        import json
        serialized = json.dumps(content, sort_keys=True)
    else:
        serialized = str(content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
