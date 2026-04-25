#!/usr/bin/env python3
"""
Qwen3-Embedding dimension-shortening wrapper for llama.cpp inference.
Receives embeddings from upstream, truncates to target dims, L2-normalizes.

Usage:
    wrapper.py <upstream-host> <upstream-port> <target-dims> [wrapper-port]
"""
import sys
import json
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

UPSTREAM_HOST = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-Embedding-4B-Q4_K_M"
UPSTREAM_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
TARGET_DIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
WRAPPER_PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 8080

class EmbeddingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silent

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Forward to upstream llama.cpp server
        try:
            upstream = urllib.request.urlopen(
                urllib.request.Request(
                    f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/v1/embeddings",
                    data=body,
                    headers={"Content-Type": "application/json"}
                ),
                timeout=60
            )
            resp = json.loads(upstream.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            out = json.dumps({"error": {"message": str(e)}}).encode()
            self.send_header("Content-Length", len(out))
            self.end_headers()
            self.wfile.write(out)
            return

        # Slice + L2-normalize
        emb = resp["data"][0]["embedding"][:TARGET_DIMS]
        norm = sum(x * x for x in emb) ** 0.5
        if norm > 0:
            emb = [x / norm for x in emb]

        resp["data"][0]["embedding"] = emb
        if "embedding_dim" in resp["data"][0]:
            resp["data"][0]["embedding_dim"] = TARGET_DIMS

        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(out))
        self.end_headers()
        self.wfile.write(out)

if __name__ == "__main__":
    print(f"Wrapper: upstream={UPSTREAM_HOST}:{UPSTREAM_PORT}, dims={TARGET_DIMS}, port={WRAPPER_PORT}")
    server = HTTPServer(("0.0.0.0", WRAPPER_PORT), EmbeddingHandler)
    server.serve_forever()
