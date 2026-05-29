"""Simple HTTP server for the benchmark dashboard."""
import http.server
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Dashboard at http://localhost:8080")
http.server.HTTPServer(('', 8080), http.server.SimpleHTTPRequestHandler).serve_forever()
