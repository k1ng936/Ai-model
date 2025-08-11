import os
import re
import io
import json
import time
import base64
import zipfile
import asyncio
import pathlib
import smtplib
import datetime
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import HTMLResponse, Response

import requests
import httpx
from bs4 import BeautifulSoup

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ======================================================================================
# App setup & config
# ======================================================================================

app = FastAPI(title="AutoGen Builder + Swarm + Research Agent")

# --- Core config ---
MODEL_NAME = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
WORKSPACE_ROOT = pathlib.Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# --- GitHub push (optional for builds/swarms) ---
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
TARGET_REPO   = os.getenv("TARGET_REPO", "")      # e.g. "yourname/yourrepo"
TARGET_BRANCH = os.getenv("TARGET_BRANCH", "main")
TARGET_DIR    = os.getenv("TARGET_DIR", "autogen-output")

# --- OpenAI key (required for agent calls) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Email notifications (SMTP) ---
SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))   # 587 for TLS
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS  = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
FROM_EMAIL    = os.getenv("FROM_EMAIL", "no-reply@your-domain.example")
TO_EMAIL      = os.getenv("TO_EMAIL", "Harpula.t@gmail.com")  # user requested

EMAIL_ENABLED = bool(SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and FROM_EMAIL and TO_EMAIL)

# --- Research agent (scanner) ---
AI_DIR_FILE     = WORKSPACE_ROOT / "ai-directory.json"
AI_SOURCES_FILE = WORKSPACE_ROOT / "ai-sources.json"
SCAN_RUNNING    = False
SCAN_CONCURRENCY = int(os.getenv("SCAN_CONCURRENCY", "8"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "3600"))  # 1 hour default

# ======================================================================================
# AutoGen agents
# ======================================================================================

model = OpenAIChatCompletionClient(model=MODEL_NAME)

# Builder: returns JSON manifest (files[], instructions, postbuild[])
builder_agent = AssistantAgent(
    "builder",
    model_client=model,
    system_message=(
        "You are a senior software engineer. "
        "When asked to 'build' an app, respond ONLY with a single JSON object:\n"
        "{\n"
        '  \"files\": [{\"path\": \"relative/path.ext\", \"content\": \"utf-8 text\"}],\n'
        '  \"instructions\": \"how to run\",\n'
        '  \"postbuild\": [\"optional shell commands\"]\n'
        "}\n"
        "No commentary. If unclear, pick sensible defaults and still return valid JSON. "
        "Use POSIX paths. No huge binaries; placeholders ok."
    ),
)

def make_worker(role_name: str) -> AssistantAgent:
    return AssistantAgent(
        role_name,
        model_client=model,
        system_message=(
            f"You are the '{role_name}' expert engineer. "
            "Write concise, implementation-focused deliverables (code, API shapes, tests). "
            "Avoid verbosity."
        ),
    )

# ======================================================================================
# Utilities
# ======================================================================================

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

def zip_folder_to_bytes(folder: pathlib.Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(folder))
    buf.seek(0)
    return buf.read()

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

async def send_email(subject: str, body: str):
    if not EMAIL_ENABLED:
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    def _send():
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            if SMTP_USE_TLS:
                s.starttls()
            s.login(SMTP_USERNAME, SMTP_PASSWORD)
            s.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())

    try:
        await asyncio.to_thread(_send)
    except Exception:
        # don't crash app on email error
        pass

# ======================================================================================
# Research Agent: fast “Quick Browser” scanner (skips CAPTCHA) + directory
# ======================================================================================

def load_json(path: pathlib.Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path: pathlib.Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def looks_like_captcha(html: str) -> bool:
    h = html.lower()
    return ("recaptcha" in h) or ("hcaptcha" in h) or ("cloudflare-turnstile" in h)

async def quick_fetch(url: str, timeout: float = 12.0) -> Optional[str]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Agent-Scanner; +https://example.invalid)"
        }) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                return None
            return r.text
    except Exception:
        return None

def extract_ai_entries(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        text = (a.get_text() or "").strip()
        if not href or len(text) < 2:
            continue
        tag = text.lower()
        # Heuristics for API/tools/docs tryouts likely free or public
        if any(k in tag for k in ["api", "docs", "documentation", "sdk", "playground", "try", "demo", "pricing", "free"]):
            try:
                url = httpx.URL(href, base=base_url).join(href).human_repr()
            except Exception:
                continue
            out.append({"name": text[:80], "url": url})
    return out[:50]

async def scan_one(url: str):
    started = time.time()
    html = await quick_fetch(url)
    if not html:
        return {"source": url, "status": "unreachable", "latency_ms": None, "items": []}
    if looks_like_captcha(html):
        return {"source": url, "status": "captcha_blocked", "latency_ms": None, "items": []}
    items = extract_ai_entries(url, html)
    latency = int((time.time() - started) * 1000)
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return {"source": url, "status": "ok", "latency_ms": latency, "items": items, "scanned_at": ts}

async def run_scan_once():
    sources = load_json(AI_SOURCES_FILE, default=[
        # You can edit via /ai/sources later. These are public directories/marketplaces.
        "https://publicapis.dev/",
        "https://rapidapi.com/hub",
        "https://www.programmableweb.com/apis/directory",
    ])
    dir_data = load_json(AI_DIR_FILE, default={"entries": [], "last_full_scan": None})
    sem = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def one(u):
        async with sem:
            return await scan_one(u)

    results = await asyncio.gather(*[one(u) for u in sources], return_exceptions=True)

    # merge & de-dup by URL
    url_seen = {e.get("url") for e in dir_data["entries"] if "url" in e}
    added = 0
    for r in results:
        if isinstance(r, Exception):
            continue
        for it in r.get("items", []):
            if it["url"] not in url_seen:
                dir_data["entries"].append({
                    "name": it["name"],
                    "url": it["url"],
                    "source": r["source"],
                    "first_seen": r.get("scanned_at"),
                    "last_seen": r.get("scanned_at"),
                    "latency_ms": r.get("latency_ms"),
                    "free_tier": True  # heuristic; manual validation later
                })
                url_seen.add(it["url"])
                added += 1
            else:
                for e in dir_data["entries"]:
                    if e.get("url") == it["url"]:
                        e["last_seen"] = r.get("scanned_at")
                        e["latency_ms"] = r.get("latency_ms")
                        break

    dir_data["last_full_scan"] = datetime.datetime.utcnow().isoformat() + "Z"
    dir_data["entries"] = dir_data["entries"][:2000]
    save_json(AI_DIR_FILE, dir_data)

    if added > 0:
        await send_email(
            subject="[Agent] Scanner update",
            body=f"Scan completed. Added {added} new entries. Total now: {len(dir_data['entries'])}."
        )
    return {"added": added, "total": len(dir_data["entries"])}

async def run_scan_loop():
    global SCAN_RUNNING
    SCAN_RUNNING = True
    while SCAN_RUNNING:
        try:
            await run_scan_once()
        except Exception:
            # swallow errors; keep loop alive
            pass
        await asyncio.sleep(SCAN_INTERVAL_SEC)

# ======================================================================================
# UI
# ======================================================================================

@app.get("/", response_class=HTMLResponse)
def ui():
    return f"""
<!doctype html>
<html><head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoGen Builder · Swarm · Research</title>
  <style>
    body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin: 20px; }}
    textarea, input {{ width: 100%; }}
    textarea {{ height: 140px; }}
    button {{ padding: 10px 14px; margin-right: 8px; margin-top: 8px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background:#f6f6f6; padding:10px; border-radius:6px; }}
    label {{ display:block; margin-top:12px; color:#444; font-weight:600; }}
    .row {{ display:flex; gap:12px; flex-wrap:wrap; }}
    .row > div {{ flex:1; min-width:220px; }}
    small{{color:#666}}
  </style>
</head>
<body>
  <h2>AutoGen Builder · Swarm · Research</h2>

  <h3>Build / Swarm</h3>
  <label>Project name</label>
  <input id="project" placeholder="e.g., project-dashboard"/>

  <label>Goal</label>
  <textarea id="goal" placeholder="e.g., Build a dashboard (frontend, FastAPI, SQLite)"></textarea>

  <div class="row">
    <div>
      <label>Swarm size</label>
      <input id="split" type="number" min="1" max="10" value="4"/>
    </div>
    <div>
      <label>Roles (comma-separated)</label>
      <input id="roles" placeholder="frontend, backend, db, qa"/>
    </div>
  </div>

  <label>Constraints / Notes (optional)</label>
  <textarea id="constraints" placeholder="Use FastAPI, plain JS, minimal CSS, include README"></textarea>

  <label><input type="checkbox" id="push"/> Push to GitHub (uses server env vars)</label>
  <div class="row">
    <div><label>Repo (env TARGET_REPO)</label><input id="repo" placeholder="{TARGET_REPO}"/></div>
    <div><label>Branch (env TARGET_BRANCH)</label><input id="branch" value="{TARGET_BRANCH}"/></div>
    <div><label>Folder (env TARGET_DIR)</label><input id="dir" value="{TARGET_DIR}"/></div>
  </div>

  <div>
    <button onclick="ask()">Ask Agent</button>
    <button onclick="build()">Build (single)</button>
    <button onclick="swarm()">Swarm (split + assemble)</button>
  </div>

  <hr/>
  <h3>Research Agent (Quick Browser)</h3>
  <p><small>Finds public/free AI tools & APIs. Skips CAPTCHAs. Benchmarks latency. Updates directory & emails summaries.</small></p>
  <div class="row">
    <div><label>Concurrency</label><input id="conc" type="number" min="1" max="64" value="{SCAN_CONCURRENCY}"/></div>
    <div><label>Interval (sec)</label><input id="interval" type="number" min="60" max="86400" value="{SCAN_INTERVAL_SEC}"/></div>
  </div>
  <div>
    <button onclick="startScan()">Start Scanner</button>
    <button onclick="stopScan()">Stop</button>
    <button onclick="statusScan()">Status</button>
    <button onclick="showDir()">Show Directory</button>
  </div>

  <pre id="out">Ready.</pre>

<script>
function out(j){{ document.getElementById('out').textContent = JSON.stringify(j,null,2); }}

async function ask(){{
  const t = document.getElementById('goal').value.trim();
  if(!t) return alert('Type a goal');
  out('Asking...');
  const r = await fetch('/agent?task=' + encodeURIComponent(t));
  out(await r.json());
}}

async function build(){{
  const payload = {{
    project: document.getElementById('project').value.trim() || 'proj-' + Date.now(),
    goal: document.getElementById('goal').value.trim(),
    push: document.getElementById('push').checked,
    repo: document.getElementById('repo').value.trim(),
    branch: document.getElementById('branch').value.trim(),
    dir: document.getElementById('dir').value.trim(),
  }};
  if(!payload.goal) return alert('Type a goal');
  out('Building...');
  const r = await fetch('/build', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
  out(await r.json());
}}

async function swarm(){{
  const roles = (document.getElementById('roles').value || '').split(',').map(s=>s.trim()).filter(Boolean);
  const payload = {{
    project: document.getElementById('project').value.trim() || 'proj-' + Date.now(),
    goal: document.getElementById('goal').value.trim(),
    split_into: parseInt(document.getElementById('split').value || '4',10),
    roles: roles,
    constraints: document.getElementById('constraints').value || '',
    push: document.getElementById('push').checked,
    repo: document.getElementById('repo').value.trim(),
    branch: document.getElementById('branch').value.trim(),
    dir: document.getElementById('dir').value.trim(),
  }};
  if(!payload.goal) return alert('Type a goal');
  out('Swarming...');
  const r = await fetch('/swarm', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
  out(await r.json());
}}

async function startScan(){{
  const conc = parseInt(document.getElementById('conc').value || '8',10);
  const interval = parseInt(document.getElementById('interval').value || '3600',10);
  const r = await fetch('/ai/start?concurrency='+conc+'&interval_sec='+interval, {{method:'POST'}});
  out(await r.json());
}}
async function stopScan(){{
  const r = await fetch('/ai/stop', {{method:'POST'}});
  out(await r.json());
}}
async function statusScan(){{
  const r = await fetch('/ai/status');
  out(await r.json());
}}
async function showDir(){{
  const r = await fetch('/ai/dir');
  out(await r.json());
}}
</script>
</body></html>
"""

# ======================================================================================
# Health & basic endpoints
# ======================================================================================

@app.get("/health")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE_ROOT)}

@app.get("/agent")
async def agent(task: str = Query(...)):
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set in environment"}
    r = await builder_agent.on_messages(
        [{"role": "user", "content": task}],
        cancellation_token=None
    )
    return {"task": task, "reply": r.messages[-1]["content"]}

# ======================================================================================
# Build (single-shot)
# ======================================================================================

@app.post("/build")
async def build(body: Dict[str, Any] = Body(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set in environment")

    project = body.get("project") or f"proj-{int(time.time())}"
    goal = body.get("goal")
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' field.")

    base = (WORKSPACE_ROOT / project)
    base.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Project: {project}\n"
        f"Goal: {goal}\n"
        "Output only the JSON manifest as described in your system message. "
        "Keep it runnable and reasonably small. Include a README.md."
    )
    res = await builder_agent.on_messages(
        [{"role": "user", "content": prompt}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]
    manifest = extract_json_manifest(text)
    files = manifest.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="No 'files' in manifest.")

    written = write_files(base, files)

    # Optional push
    pushed = False
    pushed_count = 0
    if body.get("push"):
        token = GITHUB_TOKEN or os.getenv("GITHUB_TOKEN")
        repo   = body.get("repo")   or TARGET_REPO
        branch = body.get("branch") or TARGET_BRANCH
        dir_   = body.get("dir")    or TARGET_DIR
        if not token or not repo:
            raise HTTPException(status_code=400, detail="Missing GITHUB_TOKEN or TARGET_REPO.")
        push_folder_to_github(base, repo, branch, f"{dir_}/{project}", token, f"AutoGen build: {project}")
        pushed = True
        pushed_count = len(written)

    await send_email(
        subject=f"[Agent] Build complete: {project}",
        body=f"Goal: {goal}\nFiles: {len(written)}\nPushed: {pushed}\nFolder: {base}"
    )

    return {
        "project": project,
        "workspace_path": str(base),
        "files_written": written,
        "instructions": manifest.get("instructions", ""),
        "postbuild": manifest.get("postbuild", []),
        "github": {
            "pushed": pushed,
            "repo": body.get("repo") or TARGET_REPO if pushed else None,
            "branch": body.get("branch") or TARGET_BRANCH if pushed else None,
            "path_prefix": f"{(body.get('dir') or TARGET_DIR)}/{project}" if pushed else None,
            "files_pushed": pushed_count if pushed else 0,
        },
        "download_zip": f"/download?project={project}",
        "raw_model_output_preview": text[:500],
    }

# ======================================================================================
# Swarm (coordinator → workers → assembler)
# ======================================================================================

async def run_worker(role: str, task_text: str) -> str:
    agent = make_worker(role)
    r = await agent.on_messages(
        [{"role": "user", "content": task_text}],
        cancellation_token=None
    )
    return r.messages[-1]["content"]

@app.post("/swarm")
async def swarm(body: Dict[str, Any] = Body(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    project = body.get("project") or f"proj-{int(time.time())}"
    goal = body.get("goal")
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal'")
    k = int(body.get("split_into", 4))
    roles = body.get("roles") or []
    constraints = (body.get("constraints") or "").strip()

    # Coordinator makes plan
    coordinator = AssistantAgent("coordinator", model_client=model, system_message=(
        "You are a project coordinator. Given a big goal, split it into concrete engineering subtasks "
        "according to the provided roles if available. Output a concise bullet list; each line is a deliverable."
    ))
    plan_prompt = (
        f"Project: {project}\nGoal: {goal}\nRoles: {', '.join(roles) if roles else '(not specified)'}\n"
        f"Constraints: {constraints or '(none)'}\nSplit into {k} concrete deliverables."
    )
    plan_res = await coordinator.on_messages(
        [{"role": "user", "content": plan_prompt}],
        cancellation_token=None
    )
    plan_text = plan_res.messages[-1]["content"]
    tasks = [t.strip("-• ").strip() for t in plan_text.splitlines() if t.strip()]
    tasks = [t for t in tasks if t][:k] or [goal]

    # role mapping
    if roles:
        role_for_task = [roles[i % len(roles)] for i in range(len(tasks))]
    else:
        role_for_task = [f"worker-{i+1}" for i in range(len(tasks))]

    async def run(role, task_text):
        prompt = f"{task_text}\nConstraints: {constraints}" if constraints else task_text
        return await run_worker(role, prompt)

    results = await asyncio.gather(*[run(r, t) for r, t in zip(role_for_task, tasks)], return_exceptions=True)

    # Assembler creates single manifest from worker outputs
    assembly_prompt = (
        f"Project: {project}\nGoal: {goal}\nConstraints: {constraints or '(none)'}\n\n"
        "Below are the worker results for each subtask. Using them, produce a SINGLE JSON manifest "
        "for a runnable project as per your system schema (files[], instructions, postbuild[]):\n\n"
    )
    for (t, r, rolename) in zip(tasks, results, role_for_task):
        assembly_prompt += f"### Role: {rolename}\nTask: {t}\nResult:\n{(r if not isinstance(r, Exception) else str(r))}\n\n"

    res = await builder_agent.on_messages(
        [{"role": "user", "content": assembly_prompt}],
        cancellation_token=None
    )
    text = res.messages[-1]["content"]
    manifest = extract_json_manifest(text)
    files = manifest.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="Assembler returned no 'files'.")

    base = (WORKSPACE_ROOT / project)
    base.mkdir(parents=True, exist_ok=True)
    written = write_files(base, files)

    # Optional push
    pushed = False
    pushed_count = 0
    if body.get("push"):
        token = GITHUB_TOKEN or os.getenv("GITHUB_TOKEN")
        repo   = body.get("repo")   or TARGET_REPO
        branch = body.get("branch") or TARGET_BRANCH
        dir_   = body.get("dir")    or TARGET_DIR
        if not token or not repo:
            raise HTTPException(status_code=400, detail="Missing GITHUB_TOKEN or TARGET_REPO.")
        push_folder_to_github(base, repo, branch, f"{dir_}/{project}", token, f"AutoGen swarm build: {project}")
        pushed = True
        pushed_count = len(written)

    await send_email(
        subject=f"[Agent] Swarm complete: {project}",
        body=f"Goal: {goal}\nTasks: {len(tasks)}\nFiles: {len(written)}\nPushed: {pushed}\nFolder: {base}"
    )

    return {
        "project": project,
        "goal": goal,
        "roles": role_for_task,
        "tasks_planned": tasks,
        "files_written": written,
        "instructions": manifest.get("instructions", ""),
        "postbuild": manifest.get("postbuild", []),
        "github": {
            "pushed": pushed,
            "repo": body.get("repo") or TARGET_REPO if pushed else None,
            "branch": body.get("branch") or TARGET_BRANCH if pushed else None,
            "path_prefix": f"{(body.get('dir') or TARGET_DIR)}/{project}" if pushed else None,
            "files_pushed": pushed_count if pushed else 0,
        },
        "download_zip": f"/download?project={project}",
        "raw_model_output_preview": text[:500],
    }

# ======================================================================================
# Download ZIP
# ======================================================================================

@app.get("/download")
def download(project: str = Query(...)):
    base = (WORKSPACE_ROOT / project)
    if not base.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    data = zip_folder_to_bytes(base)
    fn = f"{project}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{fn}"'}
    return Response(content=data, media_type="application/zip", headers=headers)

# ======================================================================================
# Research Agent endpoints
# ======================================================================================

@app.post("/ai/start")
async def ai_start(concurrency: int = 8, interval_sec: int = 3600):
    global SCAN_CONCURRENCY, SCAN_INTERVAL_SEC
    SCAN_CONCURRENCY = max(1, min(int(concurrency), 64))
    SCAN_INTERVAL_SEC = max(60, min(int(interval_sec), 24*3600))
    asyncio.create_task(run_scan_loop())
    return {"status": "started", "concurrency": SCAN_CONCURRENCY, "interval_sec": SCAN_INTERVAL_SEC}

@app.post("/ai/stop")
async def ai_stop():
    global SCAN_RUNNING
    SCAN_RUNNING = False
    return {"status": "stopping"}

@app.get("/ai/status")
def ai_status():
    return {
        "running": SCAN_RUNNING,
        "concurrency": SCAN_CONCURRENCY,
        "interval_sec": SCAN_INTERVAL_SEC,
        "sources_count": len(load_json(AI_SOURCES_FILE, [])),
        "entries_count": len(load_json(AI_DIR_FILE, {"entries": []})["entries"]),
    }

@app.get("/ai/dir")
def ai_dir():
    return load_json(AI_DIR_FILE, {"entries": [], "last_full_scan": None})

@app.post("/ai/sources")
def ai_sources_set(body: Dict[str, Any]):
    urls = body.get("urls", [])
    if not isinstance(urls, list) or not urls:
        raise HTTPException(status_code=400, detail="Provide 'urls': [ ... ]")
    save_json(AI_SOURCES_FILE, urls)
    return {"status": "ok", "count": len(urls)}
