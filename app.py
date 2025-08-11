import json
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

DATA_FILE = Path("projects.json")
CHAT_FILE = Path("chat.json")
app = FastAPI()

# Load saved data
def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except:
            return default
    return default

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

projects = load_json(DATA_FILE, [])
chat_log = load_json(CHAT_FILE, [])

# ---- DASHBOARD ----
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <html>
    <head>
        <title>Thomas HQ</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #fff; }
            h1 { color: #0ff; }
            input, textarea { width: 100%; padding: 10px; margin: 5px 0; }
            button { padding: 10px 20px; background: #0ff; border: none; cursor: pointer; font-weight: bold; }
            .project, .chat-msg { border: 1px solid #333; padding: 10px; margin-bottom: 10px; background: #222; }
            .done { color: #0f0; }
            .pending { color: #ff0; }
            .from-ai { color: #0ff; }
            .from-you { color: #fff; }
        </style>
    </head>
    <body>
        <h1>Thomas Hercules HQ</h1>

        <h2>Add Project</h2>
        <form action="/add" method="post">
            <input type="text" name="name" placeholder="Project Name" required>
            <textarea name="features" placeholder="List features separated by commas"></textarea>
            <button type="submit">Add Project</button>
        </form>

        <h2>Current Projects</h2>
    """
    for idx, proj in enumerate(projects):
        html += f"<div class='project'><h3>{proj['name']}</h3><ul>"
        for feat in proj["features"]:
            status = "done" if feat["done"] else "pending"
            html += f"<li class='{status}'>{feat['name']}</li>"
        html += "</ul></div>"

    html += """
        <h2>Live Chat with AI</h2>
        <form action="/chat/send" method="post">
            <input type="text" name="message" placeholder="Type your message..." required>
            <input type="hidden" name="sender" value="Thomas">
            <button type="submit">Send</button>
        </form>
        <button onclick="window.location.reload()">Refresh</button>
    """
    for msg in reversed(chat_log[-20:]):  # Show last 20 messages
        sender_class = "from-ai" if msg["sender"] == "AI" else "from-you"
        html += f"<div class='chat-msg {sender_class}'><b>{msg['sender']}:</b> {msg['text']}</div>"

    html += "</body></html>"
    return HTMLResponse(html)

# ---- PROJECT API ----
@app.post("/add")
def add_project(name: str = Form(...), features: str = Form("")):
    feature_list = [{"name": f.strip(), "done": False} for f in features.split(",") if f.strip()]
    projects.append({"name": name, "features": feature_list})
    save_json(DATA_FILE, projects)
    return HTMLResponse(f"<h2>Project '{name}' added.</h2><a href='/'>Go back</a>")

# ---- CHAT API ----
@app.post("/chat/send")
def send_message(message: str = Form(...), sender: str = Form(...)):
    chat_log.append({"sender": sender, "text": message})
    save_json(CHAT_FILE, chat_log)
    return HTMLResponse("<h2>Message sent.</h2><a href='/'>Go back</a>")

@app.get("/chat")
def get_chat():
    return JSONResponse(chat_log)
