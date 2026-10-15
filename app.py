"""Local web app: corroded RC beam assessment (residual capacity, shear, IS 456 checks).

Run:   ..\\.venv\\Scripts\\python.exe app.py          then open http://localhost:8000
       ..\\.venv\\Scripts\\python.exe app.py --port 8080

Serves only this computer (127.0.0.1). Needs the trained models from 02_train_models.py.
"""
import argparse
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from assessment import Assessor, InputError

WEB_DIR = Path(__file__).resolve().parent / "web"
ASSESSOR = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/meta":
            return self._json(200, ASSESSOR.meta())
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/assess":
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = json.loads(self.rfile.read(length) or b"{}")
            return self._json(200, ASSESSOR.assess(raw))
        except InputError as e:
            return self._json(400, {"error": str(e)})
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "Could not read the request."})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    global ASSESSOR
    p = argparse.ArgumentParser(description="Corroded RC beam assessment web app")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = p.parse_args()

    print("Loading models ...")
    ASSESSOR = Assessor()
    url = f"http://localhost:{a.port}"
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"RC corrosion assessment running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
