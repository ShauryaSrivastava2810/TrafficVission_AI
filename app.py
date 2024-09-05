import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import webbrowser

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == '/run-script':
            python_executable = sys.executable  # Get the current Python executable
            subprocess.run([python_executable, 'main.py'])
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"main.py script ran successfully!")
        elif self.path == '/open-file':
            open_file('main.py')
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"main.py file opened successfully!")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Not Found")

def open_file(filename):
    if os.name == 'nt':  # Windows
        print("Opening file")
        subprocess.run([sys.executable, filename])  # Use the current Python executable
    elif os.name == 'posix':  # Linux, MacOS, etc.
        print("Opening file")
        subprocess.run(['xdg-open', filename])

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Server running on port 8000...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
