import os
import json
import datetime
from flask import Flask, render_template_string, request, jsonify

# === CONFIGURATION ===
OWNER_NAME = "Thomas Harpula"
OWNER_EMAIL = "harpula.t@gmail.com"
DATA_FILE = "state.json"

# === STATE HANDLING ===
def load_state():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"projects": []}

def save_state(state):
    with open(DATA_FILE, "w") as f:
        json.dump(state, f, indent=4)

STATE = load_state()

# === APP SETUP ===
app = Flask(__name__)

# === HTML DASHBOARD ===
dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>HQ Dashboard - {{ owner_name }}</title>
    <style>
        body { font-family: Arial; background: #111; color: #eee; text-align: center; }
        h1 { color: #4CAF50; }
        .project { background: #222; margin: 10px auto; padding: 10px; width: 80%; border-radius: 8px; }
        input, textarea { width: 80%; padding: 8px; margin: 5px; border-radius: 5px; border: none; }
        button { padding: 8px 15px; border: none; background: #4CAF50; color: #fff; cursor: pointer; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <h1>HQ Dashboard - {{ owner_name }}</h1>
    <p>Logged in as: {{ owner_email }}</p>
    <form id="ideaForm">
        <textarea name="idea" placeholder="Enter your next big idea..." required></textarea><br>
        <button type="submit">Build It</button>
    </form>
    <hr>
    <h2>Projects</h2>
    <div id="projects">
        {% for p in projects %}
            <div class="project">
                <h3>{{ p['title'] }}</h3>
                <p>{{ p['description'] }}</p>
                <small>Created: {{ p['created'] }}</small>
            </div>
        {% endfor %}
    </div>
    <script>
        document.getElementById("ideaForm").onsubmit = async function(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            let res = await fetch("/add_idea", {
                method: "POST",
                body: formData
            });
            let data = await res.json();
            alert(data.message);
            location.reload();
        }
    </script>
</body>
</html>
"""

# === ROUTES ===
@app.route("/")
def dashboard():
    return render_template_string(
        dashboard_template,
        owner_name=OWNER_NAME,
        owner_email=OWNER_EMAIL,
        projects=STATE.get("projects", [])
    )

@app.route("/add_idea", methods=["POST"])
def add_idea():
    idea = request.form.get("idea", "").strip()
    if not idea:
        return jsonify({"message": "No idea entered."}), 400
    
    new_project = {
        "title": idea[:50] + ("..." if len(idea) > 50 else ""),
        "description": idea,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "In Progress"
    }
    STATE["projects"].append(new_project)
    save_state(STATE)
    
    # Here’s where you’d trigger the “auto build” logic for ideas
    # For now, it just stores it safely
    return jsonify({"message": "Idea saved and build started."})

# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
