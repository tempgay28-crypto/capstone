"""HTTP layer shared by the Vercel functions (api/) and the local server (serve.py)."""
import json
from http.server import BaseHTTPRequestHandler

from .assessment import Assessor, InputError

MAX_BODY = 64 * 1024
_ASSESSOR = None


def assessor():
    """Load the model once per process (a warm serverless instance reuses it)."""
    global _ASSESSOR
    if _ASSESSOR is None:
        _ASSESSOR = Assessor()
    return _ASSESSOR


def meta():
    return 200, assessor().meta()


def assess(body):
    if len(body) > MAX_BODY:
        return 413, {"error": "Request too large."}
    try:
        raw = json.loads(body or b"{}")
        if not isinstance(raw, dict):
            raise ValueError
        return 200, assessor().assess(raw)
    except InputError as e:
        return 400, {"error": str(e)}
    except (ValueError, json.JSONDecodeError):
        return 400, {"error": "Could not read the request."}


class ApiHandler(BaseHTTPRequestHandler):
    """Base request handler: subclasses route GET /api/meta and POST /api/assess."""

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(min(length, MAX_BODY + 1)) if length else b""

    def method_not_allowed(self):
        self.send_json(405, {"error": "Method not allowed."})
