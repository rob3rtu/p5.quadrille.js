import os
from flask import Flask, send_from_directory, request, Response, stream_with_context
from main import rag

current_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.abspath(os.path.join(current_dir, "../../public"))

app = Flask(__name__, static_folder=public_dir)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_hugo(path):
    full_path = os.path.join(app.static_folder, path)

    if os.path.isdir(full_path):
        return send_from_directory(full_path, 'index.html')

    if os.path.isfile(full_path):
        return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))

    if os.path.isdir(full_path + '/'):
        return send_from_directory(full_path, 'index.html')

    if os.path.isfile(full_path + '.html'):
        return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path) + '.html')

    print(f"DEBUG: 404 on path: {path} | Full path searched: {full_path}")
    return "404: Hugo page not found.", 404

@app.route('/ask', methods=['POST'])
def ask_rag():
    data = request.json
    user_query = data.get('query')
    path = data.get('url')
    
    def generate():
        for chunk in rag.ask(f"(User is currently viewing documentation for {path}) {user_query}"):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=5000)