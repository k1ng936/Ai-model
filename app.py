from flask import Flask, request, jsonify, render_template_string
import openai
import os

# === CONFIGURATION ===
AUTHORIZED_USER = "Thomas Harpula"
openai.api_key = os.getenv("OPENAI_API_KEY")  # You must set your key in environment

# === INIT ===
app = Flask(__name__)

# === HTML FRONTEND ===
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Command Center — Thomas Harpula</title>
    <style>
        body { font-family: Arial; background: #0e0e0e; color: white; padding: 2em; }
        textarea, input[type=text] { width: 100%; background: #222; color: white; border: none; padding: 1em; }
        button { padding: 1em 2em; background: #1e90ff; color: white; border: none; cursor: pointer; }
        pre { background: #111; padding: 1em; overflow-x: auto; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>🧠 Agent Command Center — Thomas Harpula</h1>
    <form method="POST">
        <input type="text" name="user" placeholder="Enter your name" required />
        <textarea name="command" rows="4" placeholder="Enter your command..." required></textarea>
        <button type="submit">Run Command</button>
    </form>
    {% if output %}
    <h2>📤 Response:</h2>
    <pre>{{ output }}</pre>
    {% endif %}
</body>
</html>
'''

# === AI EXECUTION ===
def call_openai(command):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an elite AI agent working for Thomas Harpula. Follow commands strictly."},
                {"role": "user", "content": command}
            ],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"[ERROR] {str(e)}"

# === FLASK ROUTE ===
@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    if request.method == "POST":
        user = request.form.get("user")
        command = request.form.get("command")
        if user.strip() != AUTHORIZED_USER:
            output = "❌ Unauthorized user. Only Thomas Harpula can issue commands."
        else:
            output = call_openai(command)
    return render_template_string(HTML_TEMPLATE, output=output)

# === ENTRYPOINT ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
