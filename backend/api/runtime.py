"""Safe, fail-closed server readiness, shared by discovery and submission."""

import os

from backend.data.readiness import validate_dataset
from backend.tools import data
from contracts.api import RuntimeInfo


def runtime_info() -> RuntimeInfo:
    model = os.environ.get("GEMINI_MODEL", "").strip() or None
    info = RuntimeInfo(ready=False, execution_mode="unavailable", model_id=model)
    if (
        os.environ.get("LLM_MODE", "gemini") != "gemini"
        or os.environ.get("WHACK_AGENT_TOOLS", "registry") != "registry"
        or os.environ.get("WHACK_TOOLS", "real") != "real"
    ):
        info.unavailable_reason = "Research requires LLM_MODE=gemini, WHACK_AGENT_TOOLS=registry and WHACK_TOOLS=real. Scripted demo is available separately."
        return info
    if not os.environ.get("GEMINI_API_KEY", "").strip() or not model:
        info.unavailable_reason = "Gemini credentials or model are not configured on the server."
        return info
    try:
        if int(os.environ.get("GEMINI_RPM") or "5") < 1:
            raise ValueError()
    except ValueError:
        info.unavailable_reason = "GEMINI_RPM must be a positive integer."
        return info
    try:
        manifest = validate_dataset(data.data_dir())
    except (OSError, ValueError, KeyError, TypeError):
        info.unavailable_reason = (
            "Historical dataset is missing or invalid; research is unavailable."
        )
        return info
    info.ready = True
    info.execution_mode = "gemini_real"
    info.dataset_version = manifest["dataset_version"]
    return info
