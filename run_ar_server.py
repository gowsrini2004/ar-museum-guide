"""
Simple HTTP server to serve the AR mobile demo
Run this to access the AR app on your phone
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

def run_server(port=8080):
    # Change to frontend directory
    os.chdir(os.path.join(os.path.dirname(__file__), 'frontend'))
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          AR Museum Guide - Mobile Demo Server           ║
╚══════════════════════════════════════════════════════════╝

🌐 Server running at:
   - Local: http://localhost:{port}/ar_mobile_demo.html
   
📱 To access on your phone:
   1. Make sure your phone is on the SAME WiFi network
   2. Find your computer's IP address:
      - Windows: Run 'ipconfig' and look for IPv4 Address
      - Mac/Linux: Run 'ifconfig' or 'ip addr'
   3. Open on phone: http://YOUR_IP:{port}/ar_mobile_demo.html
   
   Example: http://192.168.1.5:{port}/ar_mobile_demo.html

⚠️  Important:
   - Allow camera permissions when prompted
   - Works best in good lighting
   - Point camera at any object to see AR overlay

Press Ctrl+C to stop the server
""")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    run_server()
