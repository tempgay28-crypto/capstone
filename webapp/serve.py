"""Run the web app on this computer - same code as the Vercel deployment.

Run:   python serve.py            then open http://localhost:8000
       python serve.py --port 8080 --no-browser

Serves public/ and the two API routes (/api/meta, /api/assess) on 127.0.0.1 only.
"""
import argparse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rcassess import web

PUBLIC = Path(__file__).resolve().parent / "public"


class LocalHandler(SimpleHTTPRequestHandler, web.ApiHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/meta":
            return self.send_json(*web.meta())
        if self.path.startswith("/api/"):
            return self.method_not_allowed()
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] == "/api/assess":
            return self.send_json(*web.assess(self.read_body()))
        return self.send_json(404, {"error": "Not found."})

    def log_message(self, fmt, *args):
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser(description="Corroded RC beam assessment - local server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = p.parse_args()

    print("Loading model ...")
    web.assessor()
    url = f"http://localhost:{a.port}"
    server = ThreadingHTTPServer(("127.0.0.1", a.port), LocalHandler)
    print(f"RC corrosion assessment running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
