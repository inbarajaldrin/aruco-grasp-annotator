"""
Running-app registry for the agent endpoint pathway.

Each web app, on startup, records the port it actually bound to (ports are dynamic
via find_available_port) so an external agent / CLI can discover and reach it. The
registry is a single JSON file keyed by a slug of the app name.

This is a local single-user dev convenience, not a production service-discovery system.
The file location can be overridden with the ARUCO_AGENT_REGISTRY environment variable;
it defaults to ~/.aruco-agent/registry.json.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


def registry_path() -> Path:
    """Resolve the registry file path (env override -> default in home)."""
    env = os.environ.get("ARUCO_AGENT_REGISTRY")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".aruco-agent" / "registry.json"


def slug(app_name: str) -> str:
    """Stable, shell-friendly key from a human app name."""
    s = re.sub(r"[^a-z0-9]+", "-", app_name.lower()).strip("-")
    return s or "app"


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _read_raw() -> dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_raw(data: dict[str, Any]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic replace so concurrent readers never see a half-written file
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".registry-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def register(app_name: str, *, port: int, host: str = "127.0.0.1", pid: Optional[int] = None) -> str:
    """Record (or overwrite) a running app entry. Returns the slug key."""
    key = slug(app_name)
    data = _read_raw()
    data[key] = {
        "name": app_name,
        "slug": key,
        "host": host,
        "port": port,
        "pid": pid if pid is not None else os.getpid(),
        "started_at": time.time(),
    }
    _write_raw(data)
    return key


def unregister(app_name: str) -> None:
    """Remove an app entry (best-effort; safe if missing)."""
    key = slug(app_name)
    data = _read_raw()
    if key in data:
        del data[key]
        _write_raw(data)


def live_entries() -> dict[str, Any]:
    """Return only entries whose pid is still alive, pruning dead ones from disk."""
    data = _read_raw()
    live = {k: v for k, v in data.items() if _pid_alive(int(v.get("pid", 0)))}
    if len(live) != len(data):
        _write_raw(live)
    return live
