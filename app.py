from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from Tommy Churchill's AI Agent!"

if __name__ == '__main__':
    app.run()
