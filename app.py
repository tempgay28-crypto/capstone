"""Start the web app locally. The app itself lives in webapp/ (the folder deployed to Vercel).

Run:   ..\\.venv\\Scripts\\python.exe app.py          then open http://localhost:8000
       ..\\.venv\\Scripts\\python.exe app.py --port 8080 --no-browser
"""
import runpy
import sys
from pathlib import Path

WEBAPP = Path(__file__).resolve().parent / "webapp"
sys.path.insert(0, str(WEBAPP))
runpy.run_path(str(WEBAPP / "serve.py"), run_name="__main__")
