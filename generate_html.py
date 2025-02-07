import json
from datetime import datetime
import http.server
import socketserver
import webbrowser
from pathlib import Path

def generate_html(json_data):
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .result { 
            border: 1px solid #ddd;
            padding: 20px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .timestamp { color: #666; font-size: 0.9em; }
        .description { margin: 15px 0; line-height: 1.5; }
    </style>
</head>
<body>
    <h1>Evaluation Results</h1>
"""
    
    results = sorted(json_data, key=lambda x: x['timestamp'])
    
    for result in results:
        timestamp = datetime.fromisoformat(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        html += f"""
    <div class="result">
        <h2>{result['sequence_name']}</h2>
        <p class="timestamp">Generated at: {timestamp}</p>
        <p class="description">{result['description']}</p>
        <img src="{result['image_path']}" alt="{result['sequence_name']} sequence" style="max-width: 100%;">
    </div>
"""

    html += """
</body>
</html>
"""
    return html

def serve_directory():
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}/evaluation_results.html")
        httpd.serve_forever()

def main():
    with open('eval_results.json', 'r') as f:
        data = json.load(f)
    
    html_content = generate_html(data)
    with open('evaluation_results.html', 'w') as f:
        f.write(html_content)
    
    serve_directory()

if __name__ == "__main__":
    main()