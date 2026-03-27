"""Tiny HTTP command executor for RunPod pods. Run on pod, hit from anywhere."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess, json, urllib.parse

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        cmd = urllib.parse.unquote(self.path[1:])
        if not cmd:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Send GET /<command>")
            return
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            out = {"stdout": r.stdout, "stderr": r.stderr, "code": r.returncode}
        except Exception as e:
            out = {"error": str(e)}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())
    def log_message(self, *a): pass

print("Pod exec server on :8888")
HTTPServer(("0.0.0.0", 8888), Handler).serve_forever()
