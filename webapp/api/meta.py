"""Vercel function: GET /api/meta - form options, worked examples and model provenance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcassess import web  # noqa: E402


class handler(web.ApiHandler):
    def do_GET(self):
        self.send_json(*web.meta())

    def do_POST(self):
        self.method_not_allowed()
