from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return "Tommy Churchill's Agent Command Center is live."

@app.route('/command', methods=['POST'])
def run_command():
    data = request.get_json()

    if not data or 'command' not in data:
        return jsonify({'error': 'Missing command.'}), 400

    command = data['command'].strip().lower()

    # Simulated command processing
    response = process_command(command)

    return jsonify({'status': 'success', 'command': command, 'response': response})


def process_command(command):
    if "launch agent" in command:
        return "✅ Agent launched and ready for work."
    elif "status" in command:
        return "🟢 All systems operational. Awaiting more orders."
    elif "shutdown" in command:
        return "⚠️ Agent shutdown sequence initiated."
    else:
        return "🤖 Unknown command. Please try again."


if __name__ == '__main__':
    app.run(debug=True, port=5000)
