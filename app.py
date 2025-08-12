from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()

# Set up templates and static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dummy agent command processor
def process_command(command: str) -> str:
    command = command.strip().lower()
    if "create agent" in command:
        return "✅ Agent creation initiated. Awaiting parameters..."
    elif "status" in command:
        return "📊 All agents are reporting healthy."
    elif "start sales campaign" in command:
        return "🚀 Sales campaign activated across all agents."
    elif "shutdown" in command:
        return "🛑 All agents shutting down."
    else:
        return "🤖 Command received. Awaiting further instructions..."

# Home page: command input form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": ""})

# Command processing
@app.post("/", response_class=HTMLResponse)
async def handle_command(request: Request, command: str = Form(...)):
    response = process_command(command)
    return templates.TemplateResponse("index.html", {"request": request, "response": response, "command": command})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
