"""Vercel function: POST /api/assess - one beam in (JSON), full assessment out (JSON)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcassess import web  # noqa: E402


class handler(web.ApiHandler):
    def do_POST(self):
        self.send_json(*web.assess(self.read_body()))

    def do_GET(self):
        self.method_not_allowed()
