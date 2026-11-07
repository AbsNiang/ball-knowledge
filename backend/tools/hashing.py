"""Stable hashes of tool arguments, used for seeding and for `args_hash`."""

import hashlib
import json


def hash_args(args: dict) -> str:
    """Return a short hash that ignores key order."""
    canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
