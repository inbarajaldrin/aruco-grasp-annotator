"""
Agent endpoint CLI for the aruco-grasp-annotator web apps.

A thin client that lets an agent / human discover and trigger ANY endpoint of the
running FastAPI apps by name. Discovery is via the startup registry (which app is
running on which port) plus each app's built-in OpenAPI schema. Invocation is plain
HTTP. No app-side bridge is needed — the endpoints already are the registry.

Usage:
    aruco-agent list
    aruco-agent endpoints <app>
    aruco-agent openapi <app>
    aruco-agent call <app> <METHOD> <path> [--json '<body>'] [--query k=v ...] [--allow-motion]

`<app>` may be a registry slug or any unique prefix/substring of it.

Safety:
  - Connects to loopback only (127.0.0.1 / localhost). Pass --remote to override.
  - Motion endpoints (real robot/gripper movement) are refused unless --allow-motion
    is given, and even then require --confirm <path> to match exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from . import agent_registry

# Endpoint path fragments that command physical robot/gripper motion.
MOTION_FRAGMENTS = ("execute-grasp", "gripper-command", "move-to-safe-height", "move-home")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _fail(msg: str, code: int = 1) -> "int":
    print(f"error: {msg}", file=sys.stderr)
    return code


def _resolve_app(token: str) -> dict | None:
    """Find a live app by slug, unique prefix, or substring."""
    entries = agent_registry.live_entries()
    if token in entries:
        return entries[token]
    matches = [v for k, v in entries.items() if k.startswith(token)] or [
        v for k, v in entries.items() if token in k
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(sorted(m["slug"] for m in matches))
        print(f"error: '{token}' is ambiguous: {names}", file=sys.stderr)
    return None


def _base_url(entry: dict, allow_remote: bool) -> str | None:
    host = entry.get("host", "127.0.0.1")
    if host not in LOOPBACK_HOSTS and not allow_remote:
        print(f"error: refusing non-loopback host {host!r} (pass --remote)", file=sys.stderr)
        return None
    return f"http://{host}:{entry['port']}"


def _http(method: str, url: str, body: bytes | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, method=method.upper())
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, f"connection failed: {e.reason}"


def _fetch_openapi(base: str) -> dict | None:
    status, body = _http("GET", f"{base}/openapi.json")
    if status != 200:
        print(f"error: openapi fetch returned {status}", file=sys.stderr)
        return None
    return json.loads(body)


def cmd_list(_args) -> int:
    entries = agent_registry.live_entries()
    if not entries:
        print("(no running apps registered)")
        return 0
    print(f"{'SLUG':30} {'PORT':>6}  {'PID':>7}  NAME")
    for k in sorted(entries):
        e = entries[k]
        print(f"{k:30} {e['port']:>6}  {e['pid']:>7}  {e['name']}")
    return 0


def cmd_endpoints(args) -> int:
    entry = _resolve_app(args.app)
    if not entry:
        return _fail(f"no running app matches {args.app!r} (try: aruco-agent list)")
    base = _base_url(entry, args.remote)
    if not base:
        return 1
    spec = _fetch_openapi(base)
    if spec is None:
        return 1
    rows = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            motion = "  [MOTION]" if any(f in path for f in MOTION_FRAGMENTS) else ""
            summary = op.get("summary", "")
            rows.append((method.upper(), path, summary, motion))
    for method, path, summary, motion in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"{method:6} {path}{motion}   {summary}")
    return 0


def cmd_openapi(args) -> int:
    entry = _resolve_app(args.app)
    if not entry:
        return _fail(f"no running app matches {args.app!r}")
    base = _base_url(entry, args.remote)
    if not base:
        return 1
    spec = _fetch_openapi(base)
    if spec is None:
        return 1
    print(json.dumps(spec, indent=2))
    return 0


def cmd_call(args) -> int:
    entry = _resolve_app(args.app)
    if not entry:
        return _fail(f"no running app matches {args.app!r}")
    base = _base_url(entry, args.remote)
    if not base:
        return 1

    path = args.path if args.path.startswith("/") else "/" + args.path

    # Safety gate for physically consequential endpoints.
    if any(f in path for f in MOTION_FRAGMENTS):
        if not args.allow_motion:
            return _fail(
                f"{path} commands real robot/gripper motion. "
                f"Re-run with --allow-motion --confirm {path}"
            )
        if args.confirm != path:
            return _fail(f"motion not confirmed: pass --confirm {path}")
        print(f"[motion] proceeding with {args.method.upper()} {path}", file=sys.stderr)

    query = {}
    for kv in args.query or []:
        if "=" not in kv:
            return _fail(f"bad --query {kv!r} (expected k=v)")
        k, v = kv.split("=", 1)
        query[k] = v
    url = f"{base}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)

    body = None
    if args.json is not None:
        try:
            json.loads(args.json)  # validate
        except json.JSONDecodeError as e:
            return _fail(f"--json is not valid JSON: {e}")
        body = args.json.encode("utf-8")

    status, text = _http(args.method, url, body)
    print(f"HTTP {status}")
    try:
        print(json.dumps(json.loads(text), indent=2))
    except (json.JSONDecodeError, ValueError):
        print(text)
    return 0 if 200 <= status < 300 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aruco-agent", description="Trigger aruco web-app endpoints.")
    p.add_argument("--remote", action="store_true", help="allow non-loopback hosts")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list running apps").set_defaults(func=cmd_list)

    e = sub.add_parser("endpoints", help="list an app's endpoints")
    e.add_argument("app")
    e.set_defaults(func=cmd_endpoints)

    o = sub.add_parser("openapi", help="dump an app's raw OpenAPI schema")
    o.add_argument("app")
    o.set_defaults(func=cmd_openapi)

    c = sub.add_parser("call", help="invoke an endpoint")
    c.add_argument("app")
    c.add_argument("method")
    c.add_argument("path")
    c.add_argument("--json", help="JSON request body")
    c.add_argument("--query", action="append", help="query param k=v (repeatable)")
    c.add_argument("--allow-motion", action="store_true", help="permit robot-motion endpoints")
    c.add_argument("--confirm", help="exact path, required with --allow-motion")
    c.set_defaults(func=cmd_call)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
