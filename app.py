import os
import io
import re
import json
import time
import base64
import asyncio
import pathlib
import smtplib
import zipfile
import subprocess
import datetime
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText

from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.responses import HTMLResponse, Response

import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# =====================================================================================
# Config
# =====================================================================================

app = FastAPI(title="Head Honcho (Single Project, Static Dashboard)")

WORKSPACE = pathlib.Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
SITE_DIR = WORKSPACE / "site"
PROJECT_ROOT = WORKSPACE / "project"     # all generated features live here
STATE_FILE = WORKSPACE / "state.json"    # single-project state

for p in (WORKSPACE, SITE_DIR, PROJECT_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# Model
MODEL_NAME = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# GitHub (optional)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # PAT with repo scope
GITHUB_REPO  = os.getenv("TARGET_REPO", "")   # owner/repo (optional, can be passed per request)
GITHUB_BRANCH = os.getenv("TARGET_BRANCH", "main")
GITHUB_DIR    = os.getenv("TARGET_DIR", "autogen-output")

# Email (optional)
SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS  = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
FROM_EMAIL    = os.getenv("FROM_EMAIL", "")
TO_EMAIL      = os.getenv("TO_EMAIL", "")  # set this if you want email updates

EMAIL_ENABLED = all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL, TO_EMAIL])

# Runner (sequential by design for single project)
RUNNING = False
CURRENT_TASK_ID: Optional[str] = None

# =====================================================================================
# Agents
# =====================================================================================

model_client = OpenAIChatCompletionClient(model=MODEL_NAME)

# Builder: returns JSON manifest {files[], instructions, postbuild[]}
builder_agent = AssistantAgent(
    "builder",
    model_client=model_client,
    system_message=(
        "You are a senior software engineer. "
        "When asked to build a feature, respond ONLY with a single JSON object:\n"
        "{\n"
        '  \"files\": [{\"path\": \"relative/path.ext\", \"content\": \"utf-8 text\"}],\n'
        '  \"instructions\": \"how to run\",\n'
        '  \"postbuild\": [\"optional shell commands\"]\n'
        "}\n"
        "No commentary. Choose sensible defaults if unclear. Use POSIX paths. "
        "Avoid large binaries; placeholders ok. Include unit tests (pytest) when applicable."
    ),
)

planner_agent = AssistantAgent(
    "planner",
    model_client=model_client,
    system_message=(
        "You plan features for software projects. Given a project description, propose a concise list "
        "of concrete, testable features. Each should be deliverable with code + tests. "
        "Output a simple bullet list (one line per feature)."
    ),
)

prioritizer_agent = AssistantAgent(
    "prioritizer",
    model_client=model_client,
    system_message=(
        "You prioritize features for fastest path to a usable MVP and maximum early functionality. "
        "Given a list of features, output them reordered (top = do first). "
        "Favor items that get the app online quickly and provide core value."
    ),
)

# =====================================================================================
# State
# =====================================================================================

# STATE format:
# {
#   "project": {
#       "name": str,
#       "description": str,
#       "created_at": iso,
#       "github": {"repo": str|None, "branch": str, "dir": str},
#       "features": [
#         {"id": "short-id", "desc": "text", "status": "queued|running|done|failed",
#          "progress": int, "started_at": iso|None, "ended_at": iso|None,
#          "files": [paths], "error": str|None}
#       ],
#       "order": ["id", ...]   # prioritized order
#   }
# }

STATE: Dict[str, Any] = {"project": None}

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

def sanitize_id(s: str) -> str:
    return re.sub(r"[^\w\-]+", "-", (s or "").strip().lower())[:60].strip("-") or f"feat-{int(time.time())}"

# =====================================================================================
# Utilities
# =====================================================================================

def extract_manifest(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    raw = m.group(1) if m else text.strip()
    return json.loads(raw)

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

def gh_headers(token: str):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def gh_put(repo_full: str, branch: str, path_in_repo: str, content_text: str, token: str, message: str):
    url = f"https://api.github.com/repos/{repo_full}/contents/{path_in_repo}"
    r = requests.get(url, headers=gh_headers(token), params={"ref": branch})
    sha = r.json().get("sha") if r.status_code == 200 else None
    data = {
        "message": message,
        "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=gh_headers(token), json=data)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub push failed: {r.status_code} {r.text}")

def gh_push_folder(local_base: pathlib.Path, repo_full: str, branch: str, prefix: str, token: str, message: str):
    for p in local_base.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_base))
        dest = f"{prefix}/{rel}".strip("/")
        gh_put(repo_full, branch, dest, p.read_text(encoding="utf-8"), token, message)

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
        pass

def percent_done() -> int:
    proj = STATE.get("project")
    if not proj or not proj.get("features"):
        return 0
    vals = [int(f.get("progress", 0)) for f in proj["features"]]
    return int(sum(vals) / max(1, len(vals)))

def write_static_dashboard():
    """Regenerate a static HTML snapshot under /site/index.html."""
    proj = STATE.get("project")
    name = proj["name"] if proj else "(no project)"
    desc = proj.get("description", "") if proj else ""
    feats = proj.get("features", []) if proj else []
    order = proj.get("order", [])
    # map id -> feature
    feat_by_id = {f["id"]: f for f in feats}
    ordered_features = [feat_by_id[i] for i in order if i in feat_by_id] + [f for f in feats if f["id"] not in order]

    def tag(f):
        s = f['status']
        color = {"queued":"#999","running":"#0277bd","done":"#2e7d32","failed":"#c62828"}.get(s, "#666")
        return f"<span style='background:{color};color:white;padding:2px 6px;border-radius:10px;font-size:12px'>{s}</span>"

    rows = []
    for f in ordered_features:
        rows.append(
            f"<tr>"
            f"<td>{f['id']}</td>"
            f"<td>{f['desc']}</td>"
            f"<td>{tag(f)}</td>"
            f"<td>{f.get('progress',0)}%</td>"
            f"<td>{f.get('started_at','')}</td>"
            f"<td>{f.get('ended_at','')}</td>"
            f"</tr>"
        )
    table = "\n".join(rows) if rows else "<tr><td colspan='6'>(no features yet)</td></tr>"

    html = f"""<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} – Static Dashboard</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:20px}}
h2,h3{{margin:8px 0}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #eee;padding:8px;text-align:left;vertical-align:top}}
.progress{{background:#eee;height:10px;border-radius:6px;overflow:hidden}}
.progress>div{{background:#4caf50;height:10px;width:{percent_done()}%}}
small{{color:#666}}
</style>
</head>
<body>
<h2>Project: {name}</h2>
<p><small>Snapshot updated: {now_iso()}</small></p>
<p>{desc}</p>

<h3>Overall Progress</h3>
<div class="progress"><div></div></div>
<p><b>{percent_done()}%</b> complete</p>

<h3>Features (prioritized)</h3>
<table>
  <tr><th>ID</th><th>Description</th><th>Status</th><th>Progress</th><th>Started</th><th>Finished</th></tr>
  {table}
</table>

<p style="margin-top:20px"><a href="/control">Open Control Panel</a></p>
</body>
</html>"""
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")

# =====================================================================================
# Core worker
# =====================================================================================

async def run_next_feature():
    """Pick next queued feature by prioritized order and build it."""
    global CURRENT_TASK_ID
    proj = STATE.get("project")
    if not proj:
        return
    # find next queued following order
    feats = proj.get("features", [])
    if not feats:
        return
    order = proj.get("order", [])
    # put ordered queued first
    id_to_feat = {f["id"]: f for f in feats}
    ordered = [id_to_feat[i] for i in order if i in id_to_feat] + [f for f in feats if f["id"] not in order]
    next_feat = next((f for f in ordered if f["status"] == "queued"), None)
    if not next_feat:
        return

    # mark running
    next_feat["status"] = "running"
    next_feat["progress"] = 5
    next_feat["started_at"] = now_iso()
    CURRENT_TASK_ID = next_feat["id"]
    save_state(); write_static_dashboard()

    # ask builder
    feature_dir = (PROJECT_ROOT / next_feat["id"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        f"Project: {proj['name']}\n"
        f"Feature: {next_feat['desc']}\n"
        "Return ONLY the JSON manifest as per your system message, including unit tests (pytest) where applicable."
    )

    try:
        res = await builder_agent.on_messages(
            [{"role": "user", "content": prompt}],
            cancellation_token=None
        )
        text = res.messages[-1]["content"]
        manifest = extract_manifest(text)
        files = manifest.get("files", [])
        if not files:
            raise RuntimeError("No 'files' in manifest")
        written = write_files(feature_dir, files)
        next_feat["progress"] = 50
        next_feat.setdefault("files", [])
        next_feat["files"] = written
        save_state(); write_static_dashboard()
    except Exception as e:
        next_feat["status"] = "failed"
        next_feat["progress"] = 100
        next_feat["ended_at"] = now_iso()
        next_feat["error"] = str(e)
        save_state(); write_static_dashboard()
        await send_email(f"[HeadHoncho] Feature FAILED: {proj['name']} · {next_feat['id']}",
                         f"Error: {e}")
        CURRENT_TASK_ID = None
        return

    # run tests (best-effort)
    try:
        subprocess.run(["python", "-m", "pip", "install", "-q", "pytest"], check=False)
        proc = subprocess.run(["python", "-m", "pytest", "-q"], cwd=str(feature_dir),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        passed = (proc.returncode == 0)
        next_feat["progress"] = 85
        save_state(); write_static_dashboard()
    except Exception as e:
        passed = False
        # record error in a simple file for visibility
        (feature_dir / "_test_error.txt").write_text(str(e), encoding="utf-8")

    # mark done/failed
    if passed:
        next_feat["status"] = "done"
        next_feat["progress"] = 100
        next_feat["ended_at"] = now_iso()
    else:
        next_feat["status"] = "failed"
        next_feat["progress"] = 100
        next_feat["ended_at"] = now_iso()

    save_state(); write_static_dashboard()
    await send_email(
        f"[HeadHoncho] Feature {next_feat['status'].upper()}: {proj['name']} · {next_feat['id']}",
        f"Feature: {next_feat['desc']}\nFiles: {len(next_feat.get('files',[]))}"
    )
    CURRENT_TASK_ID = None

    # optional GitHub push if configured
    repo = (proj.get("github") or {}).get("repo") or GITHUB_REPO
    if repo and GITHUB_TOKEN:
        try:
            prefix = f"{(proj.get('github') or {}).get('dir', GITHUB_DIR)}/{proj['name']}/{next_feat['id']}"
            gh_push_folder(feature_dir, repo, (proj.get("github") or {}).get("branch", GITHUB_BRANCH),
                           prefix, GITHUB_TOKEN, f"{proj['name']}: {next_feat['id']}")
        except Exception as e:
            # ignore push errors for now; could be extended to record
            pass

async def runner_loop():
    global RUNNING
    RUNNING = True
    while RUNNING:
        # if nothing running and there's queued work, run next
        proj = STATE.get("project")
        if proj:
            running_exists = any(f["status"] == "running" for f in proj.get("features", []))
            queued_exists = any(f["status"] == "queued" for f in proj.get("features", []))
            if not running_exists and queued_exists:
                await run_next_feature()
        await asyncio.sleep(1.0)

# =====================================================================================
# Static UI (snapshot) and simple control panel
# =====================================================================================

@app.get("/", response_class=HTMLResponse)
def static_dashboard():
    idx = SITE_DIR / "index.html"
    if not idx.exists():
        write_static_dashboard()
    return HTMLResponse(idx.read_text(encoding="utf-8"))

@app.get("/control", response_class=HTMLResponse)
def control():
    proj = STATE.get("project")
    name = proj["name"] if proj else ""
    desc = proj["description"] if proj else ""
    repo = (proj.get("github") or {}).get("repo", "") if proj else (GITHUB_REPO or "")
    branch = (proj.get("github") or {}).get("branch", GITHUB_BRANCH) if proj else GITHUB_BRANCH
    dirpref = (proj.get("github") or {}).get("dir", GITHUB_DIR) if proj else GITHUB_DIR
    running = "running" if RUNNING else "stopped"
    return f"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Head Honcho Control</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:20px}}
label{{display:block;margin:8px 0 4px}}
input,textarea{{width:100%;padding:8px}}
textarea{{height:120px}}
button{{padding:10px 14px;margin:8px 8px 0 0}}
small{{color:#666}}
</style>
</head>
<body>
<h2>Head Honcho — Control Panel</h2>
<p><a href="/">View Static Dashboard</a></p>
<p><small>Runner: <b>{running}</b></small></p>

<h3>1) Define / Update Single Project</h3>
<form onsubmit="return false">
<label>Project name</label>
<input id="pname" value="{name}" placeholder="e.g., research-agent">
<label>Description</label>
<textarea id="pdesc" placeholder="Short description">{desc}</textarea>
<label>GitHub repo (owner/repo, optional)</label>
<input id="grepo" value="{repo}" placeholder="owner/repo">
<label>Branch</label>
<input id="gbranch" value="{branch}">
<label>Dir prefix</label>
<input id="gdir" value="{dirpref}">
<button onclick="saveProject()">Save Project</button>
</form>

<h3>2) Set Feature List (one per line)</h3>
<form onsubmit="return false">
<label>Features</label>
<textarea id="features" placeholder="homepage\napi endpoints\nunit tests"></textarea>
<button onclick="applyFeatures()">Apply Features</button>
<button onclick="planFeatures()">Let AI Plan From Description</button>
<button onclick="prioritize()">Prioritize (fastest online + most functionality)</button>
</form>

<h3>3) Runner</h3>
<button onclick="startRunner()">Start</button>
<button onclick="stopRunner()">Stop</button>
<button onclick="regen()">Regenerate Static Snapshot</button>

<pre id="out"></pre>

<script>
async function saveProject(){{
  const r = await fetch('/project', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{
      name: document.getElementById('pname').value.trim(),
      description: document.getElementById('pdesc').value.trim(),
      github: {{
        repo: document.getElementById('grepo').value.trim(),
        branch: document.getElementById('gbranch').value.trim(),
        dir: document.getElementById('gdir').value.trim()
      }}
    }})
  }});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}

async function applyFeatures(){{
  const txt = document.getElementById('features').value;
  const lines = txt.split(/\\n+/).map(s=>s.trim()).filter(Boolean);
  const r = await fetch('/features', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{features: lines}})
  }});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}

async function planFeatures(){{
  const r = await fetch('/plan', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}

async function prioritize(){{
  const r = await fetch('/prioritize', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}

async function startRunner(){{
  const r = await fetch('/runner/start', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}

async function stopRunner(){{
  const r = await fetch('/runner/stop', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}

async function regen(){{
  const r = await fetch('/regen', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}
</script>
</body>
</html>
"""

# =====================================================================================
# API
# =====================================================================================

@app.get("/health")
def health():
    proj_name = STATE.get("project", {}).get("name") if STATE.get("project") else None
    return {"status":"ok","project":proj_name,"running":RUNNING,"workspace":str(WORKSPACE)}

@app.post("/project")
def set_project(body: Dict[str, Any] = Body(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name required")
    desc = (body.get("description") or "").strip()
    gh = body.get("github") or {}
    STATE["project"] = {
        "name": name,
        "description": desc,
        "created_at": now_iso(),
        "github": {
            "repo": (gh.get("repo") or GITHUB_REPO) or None,
            "branch": (gh.get("branch") or GITHUB_BRANCH),
            "dir": (gh.get("dir") or GITHUB_DIR)
        },
        "features": [],
        "order": []
    }
    save_state(); write_static_dashboard()
    return {"ok": True, "project": name}

@app.post("/features")
def set_features(body: Dict[str, Any] = Body(...)):
    proj = STATE.get("project")
    if not proj:
        raise HTTPException(status_code=400, detail="Set project first")
    features_in = body.get("features")
    if not isinstance(features_in, list) or not features_in:
        raise HTTPException(status_code=400, detail="Provide features: [ ... ]")
    feats = []
    for f in features_in:
        fid = sanitize_id(f.split()[:6][0] + "-" + str(abs(hash(f)) % 10000))
        feats.append({
            "id": fid,
            "desc": f,
            "status": "queued",
            "progress": 0,
            "started_at": None,
            "ended_at": None,
            "files": [],
            "error": None
        })
    proj["features"] = feats
    proj["order"] = [x["id"] for x in feats]  # default order: input order
    save_state(); write_static_dashboard()
    return {"ok": True, "count": len(feats)}

@app.post("/plan")
async def plan_features():
    proj = STATE.get("project")
    if not proj:
        raise HTTPException(status_code=400, detail="Set project first")
    if not proj.get("description"):
        raise HTTPException(status_code=400, detail="Project description required to plan")
    res = await planner_agent.on_messages(
        [{"role":"user","content": f"Project: {proj['name']}\nDescription: {proj['description']}\nList features:"}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]
    candidates = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    if not candidates:
        raise HTTPException(status_code=400, detail="Planner returned no features")
    # replace feature list
    feats = []
    for f in candidates[:20]:
        fid = sanitize_id(f.split()[:6][0] + "-" + str(abs(hash(f)) % 10000))
        feats.append({
            "id": fid,
            "desc": f,
            "status": "queued",
            "progress": 0,
            "started_at": None,
            "ended_at": None,
            "files": [],
            "error": None
        })
    proj["features"] = feats
    proj["order"] = [x["id"] for x in feats]
    save_state(); write_static_dashboard()
    return {"ok": True, "planned": len(feats)}

@app.post("/prioritize")
async def prioritize():
    proj = STATE.get("project")
    if not proj or not proj.get("features"):
        raise HTTPException(status_code=400, detail="No features to prioritize")
    lines = "\n".join([f"- {f['desc']}" for f in proj["features"]])
    res = await prioritizer_agent.on_messages(
        [{"role":"user","content": f"Features:\n{lines}\nReorder top-to-bottom for fastest MVP and most functionality."}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]
    ordered_desc = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    # map desc -> id
    desc_to_id = {f["desc"]: f["id"] for f in proj["features"]}
    new_order = []
    for d in ordered_desc:
        if d in desc_to_id:
            new_order.append(desc_to_id[d])
    # add any missing
    for f in proj["features"]:
        if f["id"] not in new_order:
            new_order.append(f["id"])
    proj["order"] = new_order
    save_state(); write_static_dashboard()
    return {"ok": True, "order_count": len(new_order)}

@app.post("/runner/start")
async def start_runner():
    global RUNNING
    if not RUNNING:
        asyncio.create_task(runner_loop())
    write_static_dashboard()
    return {"running": True}

@app.post("/runner/stop")
async def stop_runner():
    global RUNNING
    RUNNING = False
    write_static_dashboard()
    return {"running": False}

@app.post("/regen")
def regen():
    write_static_dashboard()
    return {"ok": True}

@app.get("/status")
def status():
    proj = STATE.get("project")
    return {
        "running": RUNNING,
        "current_task": CURRENT_TASK_ID,
        "progress_percent": percent_done(),
        "project": proj or None
    }

@app.get("/download")
def download(feature_id: Optional[str] = Query(None)):
    if feature_id:
        base = PROJECT_ROOT / feature_id
        if not base.exists():
            raise HTTPException(status_code=404, detail="Feature folder not found")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in base.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(base))
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="{feature_id}.zip"'}
        return Response(content=buf.read(), media_type="application/zip", headers=headers)
    else:
        # zip entire project
        if not PROJECT_ROOT.exists():
            raise HTTPException(status_code=404, detail="No project output folder yet")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in PROJECT_ROOT.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(PROJECT_ROOT))
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="project.zip"'}
        return Response(content=buf.read(), media_type="application/zip", headers=headers)
