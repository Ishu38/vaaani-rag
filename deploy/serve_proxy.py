#!/usr/bin/env python3
"""Local dev server that mimics Vercel's same-origin proxy behavior.

Serves static files from vercel_build/. API requests matching any of the 15
prefixes are forwarded to api.vaaani.in — same as the Vercel rewrite rules.
All other paths fall back to index.html for SPA-style routing.
"""
import http.server
import urllib.request
import urllib.error
import os
from pathlib import Path

API_BASE = os.environ.get("VAAANI_API_BASE", "https://api.vaaani.in").rstrip("/")
ROOT = Path(__file__).resolve().parent / "vercel_build"

API_PREFIXES = [
    "auth", "audio", "cognitive", "feynman", "figures",
    "graph", "hermes", "ingest", "learning", "messenger",
    "simulation", "status", "youtube",
]


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        self._route()

    def do_POST(self):
        self._route()

    def do_PUT(self):
        self._route()

    def do_DELETE(self):
        self._route()

    def do_OPTIONS(self):
        self._route()

    def do_PATCH(self):
        self._route()

    def _is_api_path(self):
        if self.path == "/":
            return False
        parts = self.path.lstrip("/").split("/")
        prefix = parts[0]
        if prefix in API_PREFIXES:
            # cognitive, simulation — sub-path required (preserve .html page)
            if prefix in ("cognitive", "simulation") and len(parts) < 2:
                return False
            return True
        return False

    def _serve_static(self):
        """Serve static files with cleanUrl support (/app → app.html)."""
        path = self.path.split("?")[0]  # strip query string
        if path == "/":
            path = "/index.html"
        elif not os.path.splitext(path)[1]:
            # No extension — try clean URL: /app → app.html, /login → login.html
            clean = path + ".html"
            if (ROOT / clean.lstrip("/")).exists():
                path = clean
        self.path = path  # update path so SimpleHTTPRequestHandler can serve it
        super().do_GET()

    def _route(self):
        if self._is_api_path():
            self._proxy_to_backend()
        else:
            self._serve_static()

    def _proxy_to_backend(self):
        url = f"{API_BASE}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else None

        req = urllib.request.Request(url, data=body, method=self.command)
        # Forward relevant client headers
        for h in ("content-type", "authorization", "cookie", "accept",
                  "accept-encoding", "accept-language", "user-agent",
                  "x-forwarded-for", "x-real-ip", "x-requested-with"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                # Forward response headers (including Set-Cookie)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                # Stream the response (handles SSE)
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            try:
                self.wfile.write(e.read())
            except Exception:
                pass
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Proxy error: {e}".encode())

    def log_message(self, format, *args):
        path = args[0] if args else ""
        tag = "[API→]" if self._is_api_path() else "[static]"
        print(f"{tag} {self.command} {path}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    print(f"Serving {ROOT} on http://localhost:{port}")
    print(f"API prefixes proxied → {API_BASE}")
    http.server.HTTPServer(("", port), ProxyHandler).serve_forever()
