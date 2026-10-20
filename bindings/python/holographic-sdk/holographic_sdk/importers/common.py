# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by sources: JSON coercion, timestamps, secrets, vectors."""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

_SECRETS: set = set()


def register_secret(value: Optional[str]) -> None:
    """Remember a credential so `scrub` can mask it anywhere it might surface."""
    if value and len(value) >= 4:
        _SECRETS.add(value)


def scrub(text: str) -> str:
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, "***")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1***", text)


def redact_uri(uri: str) -> str:
    """Drop userinfo and credential-looking query values from a URI."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "<unparseable-uri>"
    if not parts.scheme and not parts.netloc:
        return uri
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port:
        host += f":{parts.port}"
    query = re.sub(r"(?i)((?:api[_-]?key|token|password|pwd|secret|key)=)[^&]*", r"\1***", parts.query)
    return urlunsplit((parts.scheme, host, parts.path, query, ""))


def secret_from(value: Optional[str], env: Dict[str, str], *env_names: str) -> Optional[str]:
    """Credential from an explicit argument, else the first set environment variable."""
    found = value
    if not found:
        for name in env_names:
            if env.get(name):
                found = env[name]
                break
    register_secret(found)
    return found


def uri_secret(uri: Optional[str]) -> Optional[str]:
    """Register the password embedded in a connection URI and return the URI."""
    if uri:
        try:
            register_secret(urlsplit(uri).password)
        except ValueError:
            pass
    return uri


# ---- JSON coercion ---------------------------------------------------------------


class Coercion(Counter):
    """Counts of lossy-but-recorded conversions made while normalizing metadata."""


def to_jsonable(value: Any, note: Optional[Coercion] = None, depth: int = 0) -> Any:
    """Convert `value` to plain JSON types. Non-finite floats and bytes are encoded
    reversibly (strings), never dropped, and each conversion is counted in `note`."""
    if depth > 64:
        if note is not None:
            note["too_deep_stringified"] += 1
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if note is not None:
            note["non_finite_float"] += 1
        return repr(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        if note is not None:
            note["bytes_base64"] += 1
        return "base64:" + base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        if isinstance(value, _dt.datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, note, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v, note, depth + 1) for v in value]
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item) and getattr(value, "shape", None) == ():
        return to_jsonable(item(), note, depth + 1)
    tolist = getattr(value, "tolist", None)  # numpy array
    if callable(tolist):
        return to_jsonable(tolist(), note, depth + 1)
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(), note, depth + 1)
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value), note, depth + 1)
    if note is not None:
        note["stringified"] += 1
    return str(value)


def parse_json_maybe(value: Any) -> Any:
    """Decode a JSON-object/array string; anything else is returned unchanged."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


# ---- timestamps ------------------------------------------------------------------


def normalize_ts(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (ISO-8601 UTC string, raw) for a timestamp. `raw` is set only when the value
    could not be parsed, so nothing is lost."""
    if value is None or value == "":
        return None, None
    try:
        if isinstance(value, bool):
            return None, str(value)
        if isinstance(value, _dt.datetime):
            dt = value
        elif isinstance(value, _dt.date):
            dt = _dt.datetime(value.year, value.month, value.day)
        elif isinstance(value, (int, float)) or (
            isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip())
        ):
            num = float(value)
            if not math.isfinite(num):
                return None, str(value)
            mag = abs(num)
            if mag >= 1e17:
                num /= 1e9
            elif mag >= 1e14:
                num /= 1e6
            elif mag >= 1e11:
                num /= 1e3
            dt = _dt.datetime.fromtimestamp(num, _dt.timezone.utc)
        elif isinstance(value, str):
            text = value.strip()
            if text.endswith(("Z", "z")):
                text = text[:-1] + "+00:00"
            dt = _dt.datetime.fromisoformat(text)
        else:
            convert = getattr(value, "to_pydatetime", None)
            if not callable(convert):
                return None, str(value)
            dt = convert()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc).isoformat(), None
    except (ValueError, OverflowError, OSError):
        return None, str(value)


# ---- vectors and iteration -------------------------------------------------------


def vector_list(vec: Any) -> Optional[List[float]]:
    """Plain list of Python floats from a list, numpy array, array.array or bytes of
    little-endian float32; `None` for missing or empty."""
    if vec is None:
        return None
    if isinstance(vec, (bytes, bytearray, memoryview)):
        import numpy as np

        out = np.frombuffer(bytes(vec), dtype="<f4").astype(np.float64).tolist()
    else:
        tolist = getattr(vec, "tolist", None)
        out = tolist() if callable(tolist) else list(vec)
    if not out:
        return None
    return [float(x) for x in out]


def chunked(it: Iterable[Any], size: int) -> Iterator[List[Any]]:
    batch: List[Any] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
