from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Agent Command Center Online. POST to /command with your command."

@app.route('/command', methods=['POST'])
def command():
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'Missing "command" in request'}), 400

        cmd = data['command'].strip().lower()
        response = handle_command(cmd)

        return jsonify({'command': cmd, 'response': response}), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


def handle_command(cmd):
    if cmd == 'launch agent':
        return '🟢 Agent launched and assigned a new task.'
    elif cmd == 'shutdown':
        return '🔴 Agent shutdown complete.'
    elif cmd == 'status':
        return '🟡 Agent status: active and waiting for commands.'
    elif cmd == 'report':
        return '📄 Last report: 3 tasks completed, 1 in progress.'
    else:
        return '🤖 Unknown command. Please try "launch agent", "shutdown", "status", or "report".'


if __name__ == '__main__':
    app.run(debug=True)
