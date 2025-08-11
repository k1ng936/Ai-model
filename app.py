import os
import io
import re
import json
import time
import base64
import asyncio
import pathlib
import smtplib
import datetime
import zipfile
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText

from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse

import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# =====================================================================================
# Config
# =====================================================================================

app = FastAPI(title="Head Coordinator — Projects, Features, Workers, Live Status")

WORKSPACE = pathlib.Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
PROJECTS_DIR = WORKSPACE / "projects"
STATE_FILE = WORKSPACE / "state.json"
for p in [WORKSPACE, PROJECTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Model for agents
MODEL_NAME = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# GitHub (optional)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # PAT with 'repo' scope to create/push
DEFAULT_BRANCH = os.getenv("TARGET_BRANCH", "main")
DEFAULT_DIR = os.getenv("TARGET_DIR", "autogen-output")

# Email (optional)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
FROM_EMAIL = os.getenv("FROM_EMAIL", "")
TO_EMAIL = os.getenv("TO_EMAIL", "")  # set to your address to receive updates

EMAIL_ENABLED = all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL, TO_EMAIL])

# =====================================================================================
# Agents
# =====================================================================================

model_client = OpenAIChatCompletionClient(model=MODEL_NAME)

# Builder agent outputs a strict JSON manifest: files[], instructions, postbuild[]
builder_agent = AssistantAgent(
    "builder",
    model_client=model_client,
    system_message=(
        "You are a senior software engineer. "
        "When asked to build a feature, respond ONLY with one JSON object:\n"
        "{\n"
        '  \"files\": [{\"path\": \"relative/path.ext\", \"content\": \"utf-8 text\"}],\n'
        '  \"instructions\": \"how to run\",\n'
        '  \"postbuild\": [\"optional shell commands\"]\n'
        "}\n"
        "No commentary. Choose sensible defaults if unclear. Use POSIX paths. "
        "Avoid large binaries (use placeholders)."
    ),
)

def make_worker(role_name: str) -> AssistantAgent:
    return AssistantAgent(
        role_name,
        model_client=model_client,
        system_message=(
            f"You are the '{role_name}' feature engineer. "
            "Write concise, implementation-focused content. Avoid verbosity."
        ),
    )

# =====================================================================================
# Persistence & Utilities
# =====================================================================================

STATE: Dict[str, Any] = {
    "projects": {},  # name -> {goal, created_at, features: {id: {...}} }
}

def load_state():
    global STATE
    if STATE_FILE.exists():
        try:
            STATE = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

def save_state():
    STATE_FILE.write_text(json.dumps(STATE, indent=2), encoding="utf-8")

load_state()

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def sanitize_name(n: str) -> str:
    return re.sub(r"[^\w\-.]+", "-", n.strip())[:80].strip("-") or f"proj-{int(time.time())}"

async def send_email(subject: str, body: str):
    if not EMAIL_ENABLED:
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    def _send():
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            if SMTP_USE_TLS:
                s.starttls()
            s.login(SMTP_USERNAME, SMTP_PASSWORD)
            s.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
    try:
        await asyncio.to_thread(_send)
    except Exception:
        pass  # do not crash on email errors

# ---------- GitHub helpers ----------

def gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def gh_user_login() -> str:
    if not GITHUB_TOKEN:
        return ""
    r = requests.get("https://api.github.com/user", headers=gh_headers())
    if r.status_code == 200:
        return r.json().get("login", "")
    return ""

def gh_create_repo_if_needed(full_or_short: str, private=False) -> str:
    """
    Returns full repo name owner/name. If short, creates/uses under PAT owner.
    """
    if "/" in full_or_short:
        full = full_or_short
        # check existence
        r = requests.get(f"https://api.github.com/repos/{full}", headers=gh_headers())
        if r.status_code in (200, 301):
            return full
        # cannot auto-create under org unless PAT has permission; skip creation here
        raise HTTPException(status_code=400, detail=f"Repo {full} not found. Create it or pass short name to create under your user.")
    # short name -> create under PAT owner
    owner = gh_user_login()
    if not owner:
        raise HTTPException(status_code=400, detail="Cannot resolve GitHub user. Check GITHUB_TOKEN.")
    # exists?
    chk = requests.get(f"https://api.github.com/repos/{owner}/{full_or_short}", headers=gh_headers())
    if chk.status_code == 200:
        return f"{owner}/{full_or_short}"
    payload = {"name": full_or_short, "private": bool(private), "auto_init": True}
    cr = requests.post("https://api.github.com/user/repos", headers=gh_headers(), json=payload)
    if cr.status_code not in (201, 202):
        raise HTTPException(status_code=400, detail=f"Create repo failed: {cr.status_code} {cr.text}")
    return f"{owner}/{full_or_short}"

def gh_put(repo_full: str, branch: str, path_in_repo: str, content_text: str, message: str):
    url = f"https://api.github.com/repos/{repo_full}/contents/{path_in_repo}"
    # get existing sha (if any)
    g = requests.get(url, headers=gh_headers(), params={"ref": branch})
    sha = g.json().get("sha") if g.status_code == 200 else None
    data = {
        "message": message,
        "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=gh_headers(), json=data)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=400, detail=f"Push failed: {r.status_code} {r.text}")

def push_folder(local_base: pathlib.Path, repo_full: str, branch: str, prefix: str, message: str):
    for p in local_base.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_base))
        dest = f"{prefix}/{rel}".strip("/")
        gh_put(repo_full, branch, dest, p.read_text(encoding="utf-8"), message)

# =====================================================================================
# Event hub (Server-Sent Events) for live dashboard
# =====================================================================================

_subscribers: List[asyncio.Queue] = []

async def publish(event: Dict[str, Any]):
    # add a timestamp and push to all subscribers
    event.setdefault("ts", now_iso())
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except Exception:
            pass

async def sse_stream():
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.append(q)
    try:
        while True:
            event = await q.get()
            data = json.dumps(event)
            yield f"data: {data}\n\n"
    finally:
        _subscribers.remove(q)

# =====================================================================================
# Build/Assemble helpers
# =====================================================================================

JSON_BLOCK = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def extract_manifest(text: str) -> Dict[str, Any]:
    m = JSON_BLOCK.search(text)
    raw = m.group(1) if m else text.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse JSON manifest: {e}")

def write_files(base: pathlib.Path, files: List[Dict[str, str]]) -> List[str]:
    written = []
    for f in files:
        rel = f.get("path")
        content = f.get("content", "")
        if not rel or ".." in rel:
            raise HTTPException(status_code=400, detail=f"Invalid path: {rel}")
        dst = (base / rel).resolve()
        if not str(dst).startswith(str(base)):
            raise HTTPException(status_code=400, detail=f"Blocked path: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        written.append(str(dst.relative_to(base)))
    return written

async def generate_feature(project: str, feature_id: str, description: str) -> Dict[str, Any]:
    """
    Ask the builder agent for code manifest for a single feature, then write files.
    """
    base = PROJECTS_DIR / project / feature_id
    base.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Project: {project}\n"
        f"Feature: {description}\n"
        "Output only the JSON manifest described in your system message (files[], instructions, postbuild[]). "
        "Keep it runnable and reasonably small. Include a README.md."
    )
    res = await builder_agent.on_messages(
        [{"role": "user", "content": prompt}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]
    manifest = extract_manifest(text)
    files = manifest.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="No 'files' in feature manifest.")
    written = write_files(base, files)
    return {"dir": str(base), "files_written": written, "manifest": manifest}

# =====================================================================================
# Coordinator: projects, features, workers, progress tracking
# =====================================================================================

# STATE["projects"][name] = {
#   "goal": str, "created_at": iso,
#   "repo": {"full": optional str, "branch": str, "dir": str} or None,
#   "features": {
#       feature_id: {
#         "desc": str,
#         "status": "queued|running|done|failed",
#         "progress": 0-100,
#         "started_at": iso|None,
#         "ended_at": iso|None,
#         "files": [..],
#         "error": str|None
#       }, ...
#   }
# }

async def run_feature_job(project: str, feature_id: str):
    feat = STATE["projects"][project]["features"][feature_id]
    feat["status"] = "running"
    feat["started_at"] = now_iso()
    feat["progress"] = 10
    save_state(); await publish({"type": "feature_start", "project": project, "feature": feature_id, "status": "running"})

    try:
        # Generate code
        result = await generate_feature(project, feature_id, feat["desc"])
        feat["progress"] = 70
        feat["files"] = result["files_written"]
        save_state(); await publish({"type": "feature_files", "project": project, "feature": feature_id, "files": feat["files"]})

        # (Optional) pseudo “tests” — here we just validate manifest presence
        # You can extend to run real tests if you add a test runner in the container.
        feat["progress"] = 90
        save_state(); await publish({"type": "feature_validated", "project": project, "feature": feature_id})

        feat["status"] = "done"
        feat["ended_at"] = now_iso()
        feat["progress"] = 100
        save_state(); await publish({"type": "feature_done", "project": project, "feature": feature_id})

        await send_email(f"[Coordinator] Feature done: {project}/{feature_id}",
                         f"Description: {feat['desc']}\nFiles: {len(feat['files'])}")

    except Exception as e:
        feat["status"] = "failed"
        feat["ended_at"] = now_iso()
        feat["error"] = str(e)
        save_state(); await publish({"type": "feature_failed", "project": project, "feature": feature_id, "error": feat["error"]})
        await send_email(f"[Coordinator] Feature FAILED: {project}/{feature_id}", f"Error: {e}")

def compute_project_progress(p: Dict[str, Any]) -> int:
    feats = p.get("features", {})
    if not feats:
        return 0
    vals = []
    for f in feats.values():
        vals.append(int(f.get("progress", 0)))
    return int(sum(vals) / max(1, len(vals)))

# =====================================================================================
# UI
# =====================================================================================

@app.get("/", response_class=HTMLResponse)
def ui():
    return f"""
<!doctype html>
<html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Head Coordinator</title>
<style>
body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin: 18px; }}
input, textarea {{ width: 100%; }}
textarea {{ height: 100px; }}
button {{ padding:10px 14px; margin:6px 8px 6px 0; }}
pre {{ background:#f6f6f6; padding:10px; border-radius:6px; white-space:pre-wrap; }}
.small {{ color:#666; }}
.row {{ display:flex; gap:12px; flex-wrap:wrap; }}
.row > div {{ flex:1; min-width:220px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #eee; padding: 6px; text-align:left; }}
.badge {{ padding:2px 8px; border-radius:999px; background:#eee; font-size:12px; }}
</style>
</head>
<body>
<h2>Head Coordinator</h2>

<h3>Create / Update Project</h3>
<div class="row">
  <div><label>Project name</label><input id="pname" placeholder="e.g., project-dashboard"/></div>
  <div><label>GitHub repo (owner/name or short to create under your user)</label><input id="repo" placeholder="yourname/my-repo or my-repo"/></div>
</div>
<div class="row">
  <div><label>Branch</label><input id="branch" value="{DEFAULT_BRANCH}"/></div>
  <div><label>Path prefix in repo</label><input id="dir" value="{DEFAULT_DIR}"/></div>
</div>
<label>Project goal</label>
<textarea id="goal" placeholder="High-level goal..."></textarea>
<button onclick="createProject()">Create/Update Project</button>

<h3>Add Feature</h3>
<div class="row">
  <div><label>Feature ID</label><input id="fid" placeholder="short-id"/></div>
</div>
<label>Feature description</label>
<textarea id="fdesc" placeholder="Exactly what to build for this feature..."></textarea>
<button onclick="addFeature()">Queue Feature</button>

<h3>Current State</h3>
<div class="small">Auto-refresh via live events.</div>
<pre id="state">Loading...</pre>

<script>
let evt;
function connect() {{
  evt = new EventSource('/events');
  evt.onmessage = (e) => {{
    try {{
      const d = JSON.parse(e.data);
      refresh();
    }} catch (err) {{}}
  }};
}}
async function refresh() {{
  const r = await fetch('/status');
  const j = await r.json();
  document.getElementById('state').textContent = JSON.stringify(j, null, 2);
}}
async function createProject(){{
  const payload = {{
    name: document.getElementById('pname').value.trim(),
    goal: document.getElementById('goal').value.trim(),
    repo: document.getElementById('repo').value.trim(),
    branch: document.getElementById('branch').value.trim(),
    dir: document.getElementById('dir').value.trim()
  }};
  const r = await fetch('/project', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
  const j = await r.json();
  alert('Project saved.');
  refresh();
}}
async function addFeature(){{
  const payload = {{
    project: document.getElementById('pname').value.trim(),
    feature_id: document.getElementById('fid').value.trim(),
    description: document.getElementById('fdesc').value.trim()
  }};
  const r = await fetch('/feature', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
  const j = await r.json();
  alert('Feature queued.');
  refresh();
}}
connect(); refresh();
</script>
</body></html>
"""

# =====================================================================================
# API: events, status, projects, features, export
# =====================================================================================

@app.get("/events")
async def events():
    return StreamingResponse(sse_stream(), media_type="text/event-stream")

@app.get("/status")
def status():
    # compute per-project progress
    out = {"projects": {}}
    for name, p in STATE["projects"].items():
        cp = dict(p)
        cp["progress"] = compute_project_progress(p)
        out["projects"][name] = cp
    return out

@app.post("/project")
async def project_save(body: Dict[str, Any] = Body(...)):
    """
    Create/update a project. If repo is given:
    - If 'owner/name' exists -> we will push into it.
    - If 'short' provided -> we will create it under your user on first export.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    name = sanitize_name(body.get("name") or "")
    if not name:
        raise HTTPException(status_code=400, detail="Missing project name")
    goal = (body.get("goal") or "").strip()
    repo = (body.get("repo") or "").strip()
    branch = (body.get("branch") or DEFAULT_BRANCH).strip()
    dir_ = (body.get("dir") or DEFAULT_DIR).strip()

    proj = STATE["projects"].get(name) or {"features": {}, "created_at": now_iso()}
    proj["goal"] = goal
    proj["repo"] = {"full": repo if repo else None, "branch": branch, "dir": dir_} if repo or proj.get("repo") else {"full": None, "branch": branch, "dir": dir_}
    STATE["projects"][name] = proj
    save_state()
    await publish({"type": "project_saved", "project": name})
    return {"ok": True, "project": name}

@app.post("/feature")
async def feature_queue(body: Dict[str, Any] = Body(...)):
    """
    Queue a feature job for a project.
    {project, feature_id, description}
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    project = sanitize_name(body.get("project") or "")
    feature_id = sanitize_name(body.get("feature_id") or "")
    desc = (body.get("description") or "").strip()
    if not (project and feature_id and desc):
        raise HTTPException(status_code=400, detail="project, feature_id, description required")
    if project not in STATE["projects"]:
        raise HTTPException(status_code=404, detail="Project not found")

    p = STATE["projects"][project]
    if "features" not in p:
        p["features"] = {}
    if feature_id in p["features"] and p["features"][feature_id]["status"] in ("queued", "running"):
        return {"ok": True, "note": "Already queued or running"}

    p["features"][feature_id] = {
        "desc": desc,
        "status": "queued",
        "progress": 0,
        "started_at": None,
        "ended_at": None,
        "files": [],
        "error": None
    }
    save_state()
    await publish({"type": "feature_queued", "project": project, "feature": feature_id})

    # fire-and-forget worker
    asyncio.create_task(run_feature_job(project, feature_id))
    return {"ok": True, "project": project, "feature": feature_id}

@app.get("/download")
def download(project: str = Query(...), feature: Optional[str] = None):
    base = PROJECTS_DIR / project
    if feature:
        base = base / feature
    if not base.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(base))
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{project + ("-" + feature if feature else "")}.zip"'}
    return Response(content=buf.read(), media_type="application/zip", headers=headers)

@app.post("/export/github")
def export_github(body: Dict[str, Any] = Body(...)):
    """
    Export a project (all features) or a single feature to GitHub.
    Body: { project, feature(optional), repo(optional), branch(optional), dir(optional), create_if_missing: bool }
    """
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=400, detail="Set GITHUB_TOKEN to export to GitHub.")

    project = sanitize_name(body.get("project") or "")
    feature = body.get("feature")
    repo_in = (body.get("repo") or "").strip()
    branch = (body.get("branch") or DEFAULT_BRANCH).strip()
    dir_ = (body.get("dir") or DEFAULT_DIR).strip()
    create_if_missing = bool(body.get("create_if_missing", True))

    if project not in STATE["projects"]:
        raise HTTPException(status_code=404, detail="Project not found")

    # resolve repo
    if repo_in:
        if create_if_missing:
            repo_full = gh_create_repo_if_needed(repo_in, private=True)
        else:
            if "/" not in repo_in:
                owner = gh_user_login()
                if not owner:
                    raise HTTPException(status_code=400, detail="Cannot resolve user for short repo name.")
                repo_full = f"{owner}/{repo_in}"
            else:
                repo_full = repo_in
    else:
        # use stored
        stored = STATE["projects"][project].get("repo") or {}
        if stored.get("full"):
            repo_full = stored["full"]
            branch = stored.get("branch", branch)
            dir_ = stored.get("dir", dir_)
        else:
            # create under user using project name
            repo_full = gh_create_repo_if_needed(project, private=True)

    # push
    if feature:
        base = PROJECTS_DIR / project / sanitize_name(feature)
        prefix = f"{dir_}/{project}/{feature}"
    else:
        base = PROJECTS_DIR / project
        prefix = f"{dir_}/{project}"

    if not base.exists():
        raise HTTPException(status_code=404, detail="Nothing to export (folder missing)")

    push_folder(base, repo_full, branch, prefix, f"Coordinator export: {project}{('/' + feature) if feature else ''}")

    asyncio.create_task(send_email("[Coordinator] Exported to GitHub",
                                   f"Pushed {project}{('/' + feature) if feature else ''} → https://github.com/{repo_full}/tree/{branch}/{prefix}"))

    return {"ok": True, "repo": repo_full, "branch": branch, "path": prefix}
