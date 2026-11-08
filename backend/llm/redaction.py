"""Remove configured credentials from persisted exception text."""

import os


def safe_error(error: Exception) -> str:
    text = repr(error)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name, "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text
