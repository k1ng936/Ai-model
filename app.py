        "You are a senior software engineer. "
   import os
import re
import json
import base64
import pathlib
import time
import asyncio
from typing import Dict, Any, List

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import HTMLResponse
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import requests

app = FastAPI(title="AutoGen Builder & Swarm")

# ========= Config =========
MODEL_NAME = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
WORKSPACE_ROOT = pathlib.Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# Optional GitHub push target (for /build)
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
TARGET_REPO   = os.getenv("TARGET_REPO", "")      # e.g. "yourname/yourrepo"
TARGET_BRANCH = os.getenv("TARGET_BRANCH", "main")
TARGET_DIR    = os.getenv("TARGET_DIR", "autogen-output")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ========= AutoGen =========
model = OpenAIChatCompletionClient(model=MODEL_NAME)
assistant = AssistantAgent(
    "builder",
    model_client=model,
    system_message=(
        "You are a senior software engineer. "
        "When asked to 'build' an app, respond ONLY with a single JSON object:\n"
        "{\n"
        '  "files": [{"path": "relative/path.ext", "content": "utf-8 text"}],\n'
        '  "instructions": "how to run",\n'
        '  "postbuild": ["optional shell commands"]\n'
        "}\n"
        "No commentary. If unclear, pick sensible defaults and still return valid JSON. "
        "Use POSIX paths. No huge binaries; placeholders ok."
    ),
)

# ========= Helpers =========
JSON_BLOCK = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def extract_json_manifest(text: str) -> Dict[str, Any]:
    m = JSON_BLOCK.search(text)
    raw = m.group(1) if m else text.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse JSON manifest: {e}")

def write_files(base_dir: pathlib.Path, files: List[Dict[str, str]]) -> List[str]:
    written = []
    for f in files:
        rel = f.get("path")
        content = f.get("content", "")
        if not rel or ".." in rel:
            raise HTTPException(status_code=400, detail=f"Invalid path: {rel}")
        dst = (base_dir / rel).resolve()
        if not str(dst).startswith(str(base_dir)):
            raise HTTPException(status_code=400, detail=f"Blocked path: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        written.append(str(dst.relative_to(base_dir)))
    return written

def github_put_file(repo: str, branch: str, path_in_repo: str, content_text: str, token: str, message: str):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    sha = None
    r = requests.get(url, headers=headers, params={"ref": branch})
    if r.status_code == 200:
        sha = r.json().get("sha")
    data = {
        "message": message,
        "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=headers, json=data)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=400, detail=f"GitHub push failed: {r.status_code} {r.text}")

def push_folder_to_github(local_base: pathlib.Path, repo: str, branch: str, target_dir: str, token: str, message: str):
    for p in local_base.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_base))
        dest = f"{target_dir}/{rel}".strip("/")
        github_put_file(repo, branch, dest, p.read_text(encoding="utf-8"), token, message)

async def run_worker(task_text: str) -> str:
    agent = AssistantAgent("worker", model_client=model)
    r = await agent.on_messages(
        [{"role": "user", "content": task_text}],
        cancellation_token=None
    )
    return r.messages[-1]["content"]

# ========= UI / Health =========
@app.get("/health")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE_ROOT)}

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoGen Builder & Swarm</title>
  <style>
    body { font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin: 20px; }
    textarea, input { width: 100%; }
    textarea { height: 140px; }
    button { padding: 10px 14px; margin-right: 8px; margin-top: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; background:#f6f6f6; padding:10px; border-radius:6px; }
    label { display:block; margin-top:10px; color:#444; }
  </style>
</head>
<body>
  <h2>AutoGen Builder & Swarm</h2>
  <p>Type a request (Ask), a build goal (Build), or split the goal among multiple workers (Swarm).</p>
  <label>Text</label>
  <textarea id="text" placeholder="e.g., Build a FastAPI TODO API with SQLite"></textarea>
  <label>Swarm size</label>
  <input id="split" type="number" min="1" max="10" value="4"/>
  <div>
    <button onclick="ask()">Ask Agent</button>
    <button onclick="build()">Build App</button>
    <button onclick="swarm()">Swarm</button>
  </div>
  <pre id="out">Ready.</pre>
<script>
async function ask(){
  const t = document.getElementById('text').value.trim();
  if(!t){ alert('Type something first'); return; }
  document.getElementById('out').textContent = 'Asking...';
  const r = await fetch('/agent?task=' + encodeURIComponent(t));
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
async function build(){
  const t = document.getElementById('text').value.trim();
  if(!t){ alert('Type a build goal first'); return; }
  document.getElementById('out').textContent = 'Building... (this can take a bit)';
  const r = await fetch('/build', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({goal: t})
  });
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
async function swarm(){
  const t = document.getElementById('text').value.trim();
  const k = parseInt(document.getElementById('split').value || '4', 10);
  if(!t){ alert('Type a goal first'); return; }
  document.getElementById('out').textContent = 'Swarming...';
  const r = await fetch('/swarm', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({goal: t, split_into: k})
  });
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
"""

# ========= Endpoints =========
@app.get("/agent")
async def agent(task: str = Query(..., description="What you want AutoGen to do")):
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set in environment"}
    res = await assistant.on_messages(
        [{"role": "user", "content": task}],
        cancellation_token=None
    )
    return {"task": task, "reply": res.messages[-1]["content"]}

@app.post("/build")
async def build(body: Dict[str, Any] = Body(..., example={"goal": "Build a FastAPI + SQLite TODO API with CRUD"})):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set in environment")

    goal = body.get("goal")
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' field.")

    session_id = f"proj-{int(time.time())}"
    base = (WORKSPACE_ROOT / session_id)
    base.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Goal: {goal}\n"
        "Output only the JSON manifest as described in your system message. "
        "Keep it runnable and reasonably small. Include a README.md."
    )
    res = await assistant.on_messages(
        [{"role": "user", "content": prompt}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]

    manifest = extract_json_manifest(text)
    files = manifest.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="No 'files' in manifest.")

    written = write_files(base, files)

    pushed = False
    pushed_count = 0
    if GITHUB_TOKEN and TARGET_REPO:
        push_folder_to_github(
            local_base=base,
            repo=TARGET_REPO,
            branch=TARGET_BRANCH,
            target_dir=f"{TARGET_DIR}/{session_id}",
            token=GITHUB_TOKEN,
            message=f"AutoGen build: {goal}",
        )
        pushed = True
        pushed_count = len(written)

    return {
        "goal": goal,
        "session": session_id,
        "workspace_path": str(base),
        "files_written": written,
        "instructions": manifest.get("instructions", ""),
        "postbuild": manifest.get("postbuild", []),
        "github": {
            "pushed": pushed,
            "repo": TARGET_REPO,
            "branch": TARGET_BRANCH,
            "path_prefix": f"{TARGET_DIR}/{session_id}" if pushed else None,
            "files_pushed": pushed_count if pushed else 0,
        },
        "raw_model_output_preview": text[:500],
    }

@app.post("/swarm")
async def swarm(body: Dict[str, Any] = Body(..., example={"goal": "Build landing page + API + tests", "split_into": 4})):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    goal = body.get("goal")
    k = int(body.get("split_into", 4))
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal'")

    # Coordinator proposes concrete sub-tasks
    coordinator = AssistantAgent("coordinator", model_client=model)
    plan_prompt = (
        f"Big goal: {goal}\n"
        f"Split into {k} concrete developer sub-tasks (bullet list). "
        "Each sub-task must be self-contained and deliverable."
    )
    plan_res = await coordinator.on_messages(
        [{"role": "user", "content": plan_prompt}],
        cancellation_token=None
    )
    plan_text = plan_res.messages[-1]["content"]
    tasks = [t.strip("-• ").strip() for t in plan_text.splitlines() if t.strip()]
    tasks = [t for t in tasks if t][:k] or [goal]

    # Run workers concurrently
    results = await asyncio.gather(*[run_worker(t) for t in tasks], return_exceptions=True)

    combined = []
    for t, r in zip(tasks, results):
        if isinstance(r, Exception):
            combined.append({"task": t, "error": str(r)})
        else:
            combined.append({"task": t, "result": r})

    return {
        "goal": goal,
        "tasks_planned": tasks,
        "results": combined
    }     "When asked to 'build' an app, respond ONLY with a single JSON object in this schema:\n"
        "{\n"
        '  \"files\": [{\"path\": \"relative/path.ext\", \"content\": \"file content as UTF-8 text\"}],\n'
        '  \"instructions\": \"how to run\",\n'
        '  \"postbuild\": [\"optional shell commands\"]\n'
        "}\n"
        "Do not include commentary. If the request is unclear, choose sensible defaults and still return valid JSON. "
        "Keep paths POSIX-style. Avoid huge binaries; use placeholders where needed."
    ),
)

# ========= Helpers =========
JSON_BLOCK = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def extract_json_manifest(text: str) -> Dict[str, Any]:
    """Find a JSON code block or use the last JSON-looking object."""
    m = JSON_BLOCK.search(text)
    raw = m.group(1) if m else text.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse JSON manifest: {e}")

def write_files(base_dir: pathlib.Path, files: List[Dict[str, str]]) -> List[str]:
    written = []
    for f in files:
        rel = f.get("path")
        content = f.get("content", "")
        if not rel or ".." in rel:
            raise HTTPException(status_code=400, detail=f"Invalid path: {rel}")
        dst = (base_dir / rel).resolve()
        if not str(dst).startswith(str(base_dir)):
            raise HTTPException(status_code=400, detail=f"Blocked path: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        written.append(str(dst.relative_to(base_dir)))
    return written

def github_put_file(repo: str, branch: str, path_in_repo: str, content_text: str, token: str, message: str):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    # get existing sha if file exists
    sha = None
    r = requests.get(url, headers=headers, params={"ref": branch})
    if r.status_code == 200:
        sha = r.json().get("sha")

    data = {
        "message": message,
        "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, json=data)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=400, detail=f"GitHub push failed: {r.status_code} {r.text}")

def push_folder_to_github(local_base: pathlib.Path, repo: str, branch: str, target_dir: str, token: str, message: str):
    for p in local_base.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_base))
        dest = f"{target_dir}/{rel}".strip("/")
        github_put_file(repo, branch, dest, p.read_text(encoding="utf-8"), token, message)

# ========= UI / Health =========
@app.get("/health")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE_ROOT)}

@app.get("/", response_class=HTMLResponse)
def ui():
    # minimal mobile-friendly UI
    return """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoGen Builder</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
    textarea { width: 100%; height: 140px; }
    button { padding: 10px 14px; margin-right: 8px; margin-top: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; background:#f6f6f6; padding:10px; border-radius:6px; }
    .hint { color:#555; font-size: 0.9em; }
  </style>
</head>
<body>
  <h2>AutoGen Builder</h2>
  <p class="hint">Type what you want (e.g., “Build a FastAPI TODO API with SQLite”).</p>
  <textarea id="text" placeholder="Your goal or question..."></textarea>
  <div>
    <button onclick="ask()">Ask Agent</button>
    <button onclick="build()">Build App</button>
  </div>
  <pre id="out">Ready.</pre>

<script>
async function ask(){
  const t = document.getElementById('text').value.trim();
  if(!t){ alert('Type something first'); return; }
  document.getElementById('out').textContent = 'Asking...';
  const r = await fetch('/agent?task=' + encodeURIComponent(t));
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
async function build(){
  const t = document.getElementById('text').value.trim();
  if(!t){ alert('Type a build goal first'); return; }
  document.getElementById('out').textContent = 'Building... (this can take a bit)';
  const r = await fetch('/build', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({goal: t})
  });
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
"""

# ========= Endpoints =========
@app.get("/agent")
async def agent(task: str = Query(..., description="What you want AutoGen to do")):
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set in environment"}
    res = await assistant.on_messages(
        [{"role": "user", "content": task}],
        cancellation_token=None
    )
    return {"task": task, "reply": res.messages[-1]["content"]}

@app.post("/build")
async def build(body: Dict[str, Any] = Body(..., example={"goal": "Build a FastAPI + SQLite TODO API with CRUD"})):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set in environment")

    goal = body.get("goal")
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' field.")

    session_id = f"proj-{int(time.time())}"
    base = (WORKSPACE_ROOT / session_id)
    base.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Goal: {goal}\n"
        "Output only the JSON manifest as described in your system message. "
        "Keep it reasonably small and runnable. Include a README.md in files."
    )
    res = await assistant.on_messages(
        [{"role": "user", "content": prompt}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]

    manifest = extract_json_manifest(text)
    files = manifest.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="No 'files' in manifest.")

    written = write_files(base, files)

    # Optional GitHub push
    pushed = False
    pushed_count = 0
    if GITHUB_TOKEN and TARGET_REPO:
        push_folder_to_github(
            local_base=base,
            repo=TARGET_REPO,
            branch=TARGET_BRANCH,
            target_dir=f"{TARGET_DIR}/{session_id}",
            token=GITHUB_TOKEN,
            message=f"AutoGen build: {goal}",
        )
        pushed = True
        pushed_count = len(written)

    return {
        "goal": goal,
        "session": session_id,
        "workspace_path": str(base),
        "files_written": written,
        "instructions": manifest.get("instructions", ""),
        "postbuild": manifest.get("postbuild", []),
        "github": {
            "pushed": pushed,
            "repo": TARGET_REPO,
            "branch": TARGET_BRANCH,
            "path_prefix": f"{TARGET_DIR}/{session_id}" if pushed else None,
            "files_pushed": pushed_count if pushed else 0,
        },
        "raw_model_output_preview": text[:500],
    }
