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
from fastapi.responses import HTMLResponse, Response, PlainTextResponse

import requests

# ---- Optional AutoGen imports with graceful fallback ----
AUTOGEN_OK = True
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except Exception:
    AUTOGEN_OK = False

# =====================================================================================
# Config
# =====================================================================================

app = FastAPI(title="Head of the Snake · Single Project · Static Dashboard · GitHub Checkpoint")

WORKSPACE = pathlib.Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
SITE_DIR = WORKSPACE / "site"
PROJECT_ROOT = WORKSPACE / "project"     # generated files live here
STATE_FILE = WORKSPACE / "state.json"    # single-project state
LOG_DIR = WORKSPACE / "logs"

for p in (WORKSPACE, SITE_DIR, PROJECT_ROOT, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Model / keys
MODEL_NAME = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# GitHub (checkpoint + issues)
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")   # PAT with 'repo' scope
GITHUB_REPO   = os.getenv("TARGET_REPO", "")    # owner/repo to store state & issues (optional but recommended)
GITHUB_BRANCH = os.getenv("TARGET_BRANCH", "main")
GITHUB_DIR    = os.getenv("TARGET_DIR", "autogen-output")  # folder prefix inside repo
GH_AUTORESTORE = os.getenv("GH_AUTORESTORE", "true").lower() == "true"  # pull state on boot if possible

# Email (optional)
SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS  = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
FROM_EMAIL    = os.getenv("FROM_EMAIL", "")
TO_EMAIL      = os.getenv("TO_EMAIL", "")

EMAIL_ENABLED = all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL, TO_EMAIL])

# Runner flags
RUNNING = False
CURRENT_FEATURE_ID: Optional[str] = None

# Auto-fix controls
MAX_FIX_ATTEMPTS_PER_FEATURE = 1  # try one automated fix cycle per feature

# =====================================================================================
# Safe helpers
# =====================================================================================

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def percent(val: float) -> int:
    try:
        v = float(val)
        return max(0, min(100, int(round(v))))
    except Exception:
        return 0

def sanitize_id(s: str) -> str:
    return re.sub(r"[^\w\-]+", "-", (s or "").strip().lower())[:60].strip("-") or f"feat-{int(time.time())}"

def read_json(path: pathlib.Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def write_json(path: pathlib.Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# State shape:
# STATE = {
#   "project": {
#       "name": str,
#       "description": str,
#       "created_at": iso,
#       "github": {"repo": str|None, "branch": str, "dir": str},
#       "features": [
#           {"id": str, "desc": str, "status": "queued|running|done|failed",
#            "progress": int, "started_at": iso|None, "ended_at": iso|None,
#            "files": [paths], "error": str|None, "fix_steps": [..], "tries": int}
#       ],
#       "order": ["feature-id", ...]
#   }
# }

STATE: Dict[str, Any] = read_json(STATE_FILE, {"project": None})

def save_state():
    write_json(STATE_FILE, STATE)

# =====================================================================================
# Email (optional)
# =====================================================================================

async def send_email(subject: str, body: str):
    if not EMAIL_ENABLED:
        return
    try:
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

        await asyncio.to_thread(_send)
    except Exception:
        pass

# =====================================================================================
# GitHub: contents + issues (task tracking) + checkpoint/restore
# =====================================================================================

def gh_headers(token: str = None):
    token = token if token is not None else GITHUB_TOKEN
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def gh_put_file(repo_full: str, branch: str, path_in_repo: str, content_text: str, message: str, token: str = None):
    url = f"https://api.github.com/repos/{repo_full}/contents/{path_in_repo}"
    try:
        r = requests.get(url, headers=gh_headers(token), params={"ref": branch}, timeout=20)
        sha = r.json().get("sha") if r.status_code == 200 else None
        data = {
            "message": message,
            "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }
        if sha:
            data["sha"] = sha
        r = requests.put(url, headers=gh_headers(token), json=data, timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"GitHub put failed: {r.status_code} {r.text}")
    except Exception as e:
        raise RuntimeError(f"GitHub put error: {e}")

def gh_get_file(repo_full: str, branch: str, path_in_repo: str, token: str = None) -> Optional[str]:
    url = f"https://api.github.com/repos/{repo_full}/contents/{path_in_repo}"
    try:
        r = requests.get(url, headers=gh_headers(token), params={"ref": branch}, timeout=20)
        if r.status_code != 200:
            return None
        enc = r.json().get("content")
        if not enc:
            return None
        return base64.b64decode(enc).decode("utf-8")
    except Exception:
        return None

def gh_push_folder(local_base: pathlib.Path, repo_full: str, branch: str, prefix: str, message: str, token: str = None):
    for p in local_base.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_base))
        dest = f"{prefix}/{rel}".strip("/")
        gh_put_file(repo_full, branch, dest, p.read_text(encoding="utf-8"), message, token)

# ---- Issues for each feature (task tracking you can continue on GitHub) ----

def gh_create_issue(repo_full: str, title: str, body: str, labels: List[str]) -> Optional[int]:
    if not GITHUB_TOKEN or not repo_full:
        return None
    try:
        url = f"https://api.github.com/repos/{repo_full}/issues"
        r = requests.post(url, headers=gh_headers(), json={"title": title, "body": body, "labels": labels}, timeout=20)
        if r.status_code in (201, 202):
            return r.json().get("number")
        return None
    except Exception:
        return None

def gh_comment_issue(repo_full: str, issue_number: int, body: str):
    if not GITHUB_TOKEN or not repo_full or not issue_number:
        return
    try:
        url = f"https://api.github.com/repos/{repo_full}/issues/{issue_number}/comments"
        requests.post(url, headers=gh_headers(), json={"body": body}, timeout=20)
    except Exception:
        pass

def gh_update_issue_labels(repo_full: str, issue_number: int, labels: List[str]):
    if not GITHUB_TOKEN or not repo_full or not issue_number:
        return
    try:
        url = f"https://api.github.com/repos/{repo_full}/issues/{issue_number}"
        requests.patch(url, headers=gh_headers(), json={"labels": labels}, timeout=20)
    except Exception:
        pass

# ---- Checkpoint state + artifacts to GitHub so next server can pick up where left off ----

def checkpoint_to_github(note: str = ""):
    proj = STATE.get("project")
    if not proj:
        return
    repo = (proj.get("github") or {}).get("repo") or GITHUB_REPO
    if not repo or not GITHUB_TOKEN:
        return
    branch = (proj.get("github") or {}).get("branch", GITHUB_BRANCH)
    dirpref = (proj.get("github") or {}).get("dir", GITHUB_DIR)

    # push state.json
    path_state = f"{dirpref}/{proj['name']}/state/state.json"
    gh_put_file(repo, branch, path_state, json.dumps(STATE, indent=2), f"[checkpoint] {proj['name']} {note}")

    # push generated files per feature
    for f in proj.get("features", []):
        fdir = PROJECT_ROOT / f["id"]
        if fdir.exists():
            prefix = f"{dirpref}/{proj['name']}/features/{f['id']}"
            try:
                gh_push_folder(fdir, repo, branch, prefix, f"[artifacts] {proj['name']} · {f['id']}")
            except Exception:
                pass

def restore_from_github() -> bool:
    proj = STATE.get("project")
    # If no project yet, we cannot know where to pull state from; skip.
    repo = (proj.get("github") or {}).get("repo") if proj else (GITHUB_REPO or "")
    if not repo or not GITHUB_TOKEN:
        return False
    branch = (proj.get("github") or {}).get("branch", GITHUB_BRANCH if proj else GITHUB_BRANCH)
    dirpref = (proj.get("github") or {}).get("dir", GITHUB_DIR if proj else GITHUB_DIR)

    # attempt to read state
    # if no project name yet, try to discover under dirpref: skip for simplicity — require project set first then restore.
    if not proj:
        return False
    path_state = f"{dirpref}/{proj['name']}/state/state.json"
    remote = gh_get_file(repo, branch, path_state)
    if not remote:
        return False
    try:
        restored = json.loads(remote)
        # merge: prefer restored if it has features
        if restored.get("project"):
            # write down restored to local, but keep any local fields that might have been set (minimal merge)
            STATE["project"] = restored["project"]
            save_state()
            return True
    except Exception:
        return False
    return False

# =====================================================================================
# Model / Agents (with graceful fallback)
# =====================================================================================

if AUTOGEN_OK:
    model_client = OpenAIChatCompletionClient(model=MODEL_NAME)

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

    diagnoser_agent = AssistantAgent(
        "diagnoser",
        model_client=model_client,
        system_message=(
            "You analyze build/test errors and produce CLEAR, step-by-step fix instructions for a developer. "
            "Given: feature description, file list, error logs. Output a short numbered list of steps the user can take."
        ),
    )

    fixer_agent = AssistantAgent(
        "fixer",
        model_client=model_client,
        system_message=(
            "You are an automatic code fixer. "
            "Return ONLY a JSON manifest of minimal file replacements to fix the described errors:\n"
            "{ \"files\": [ {\"path\": \"...\", \"content\": \"...\"} ] }\n"
            "No commentary. Only include files that must change."
        ),
    )
else:
    builder_agent = planner_agent = prioritizer_agent = diagnoser_agent = fixer_agent = None

# =====================================================================================
# HTML snapshot (static) — never crashes
# =====================================================================================

def overall_progress() -> int:
    proj = STATE.get("project")
    if not proj or not proj.get("features"):
        return 0
    vals = [percent(f.get("progress", 0)) for f in proj["features"]]
    return percent(sum(vals) / max(1, len(vals)))

def write_static_dashboard():
    proj = STATE.get("project")
    name = proj["name"] if proj else "(no project)"
    desc = proj.get("description", "") if proj else ""
    feats = proj.get("features", []) if proj else []
    order = proj.get("order", [])
    by_id = {f["id"]: f for f in feats}
    ordered = [by_id[i] for i in order if i in by_id] + [f for f in feats if f["id"] not in order]

    def tag(s: str) -> str:
        color = {"queued": "#888", "running": "#1976d2", "done": "#2e7d32", "failed": "#c62828"}.get(s, "#666")
        return f"<span style='background:{color};color:#fff;padding:2px 6px;border-radius:10px;font-size:12px'>{s}</span>"

    rows = []
    for f in ordered:
        fix_html = ""
        if f.get("fix_steps"):
            fix_html = "<ul>" + "".join([f"<li>{step}</li>" for step in f["fix_steps"][:6]]) + "</ul>"
        err = f.get("error", "")
        rows.append(
            "<tr>"
            f"<td>{f.get('id','')}</td>"
            f"<td>{f.get('desc','')}</td>"
            f"<td>{tag(f.get('status','queued'))}</td>"
            f"<td>{percent(f.get('progress',0))}%</td>"
            f"<td>{f.get('started_at','')}</td>"
            f"<td>{f.get('ended_at','')}</td>"
            f"</tr>"
            + (f"<tr><td colspan='6'><b>Errors:</b> <pre style='white-space:pre-wrap'>{err[:2000]}</pre>"
               f"{('<b>How to fix:</b>' + fix_html) if fix_html else ''}</td></tr>" if (err or fix_html) else "")
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
.progress>div{{background:#4caf50;height:10px;width:{overall_progress()}%}}
small{{color:#666}}
a.button{{display:inline-block;margin-top:10px;padding:10px 14px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none}}
</style>
</head>
<body>
<h2>Project: {name}</h2>
<p><small>Snapshot updated: {now_iso()}</small></p>
<p>{desc}</p>

<h3>Overall Progress</h3>
<div class="progress"><div></div></div>
<p><b>{overall_progress()}%</b> complete</p>

<h3>Features (prioritized)</h3>
<table>
  <tr><th>ID</th><th>Description</th><th>Status</th><th>Progress</th><th>Started</th><th>Finished</th></tr>
  {table}
</table>

<p><a class="button" href="/control">Open Control Panel</a></p>
</body>
</html>"""
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")

if not (SITE_DIR / "index.html").exists():
    write_static_dashboard()

# =====================================================================================
# Manifest + file utilities
# =====================================================================================

def extract_manifest(text: str) -> Dict[str, Any]:
    if not text:
        raise HTTPException(status_code=400, detail="Empty model response")
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    raw = (m.group(1) if m else text).strip()
    try:
        return json.loads(raw)
    except Exception:
        m2 = re.search(r"(\{(?:[^{}]|(?1))*\})", raw, re.DOTALL)
        if not m2:
            raise HTTPException(status_code=400, detail="Could not find JSON manifest in model output")
        return json.loads(m2.group(1))

def write_files(base: pathlib.Path, files: List[Dict[str, str]]) -> List[str]:
    written = []
    for f in files or []:
        rel = f.get("path")
        content = f.get("content", "")
        if not rel or ".." in rel:
            raise HTTPException(status_code=400, detail=f"Invalid path in manifest: {rel!r}")
        dst = (base / rel).resolve()
        if not str(dst).startswith(str(base)):
            raise HTTPException(status_code=400, detail=f"Blocked path: {rel!r}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        written.append(str(dst.relative_to(base)))
    return written

# =====================================================================================
# Build Pipeline (with auto-fix attempt & GitHub issues)
# =====================================================================================

async def run_pytests(folder: pathlib.Path) -> (bool, str):
    try:
        subprocess.run(["python", "-m", "pip", "install", "-q", "pytest"], check=False)
        proc = subprocess.run(["python", "-m", "pytest", "-q"], cwd=str(folder),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
        return (proc.returncode == 0), proc.stdout
    except Exception as e:
        return False, f"(pytest error) {e}"

async def diagnose_and_fix(feature: Dict[str, Any], proj: Dict[str, Any], test_output: str) -> (bool, str, List[Dict[str,str]]):
    """Ask diagnoser for steps; ask fixer for minimal patch manifest; return (fixed?, steps_text, files_to_write)."""
    steps: List[str] = []
    fix_files: List[Dict[str, str]] = []
    # 1) Diagnoser: steps for human & dashboard
    if AUTOGEN_OK and diagnoser_agent:
        try:
            files_preview = "\n".join(feature.get("files", [])[:20])
            diag_prompt = (
                f"Project: {proj['name']}\n"
                f"Feature: {feature['desc']}\n"
                f"Files:\n{files_preview}\n\n"
                f"Error logs / pytest output (tail):\n{test_output[-2000:]}\n\n"
                "Provide a short numbered list of concrete steps to fix the problem."
            )
            res = await diagnoser_agent.on_messages(
                [{"role": "user", "content": diag_prompt}],
                cancellation_token=None
            )
            text = (res.messages[-1]["content"] or "").strip()
            steps = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
        except Exception:
            steps = []
    # 2) Fixer: minimal patch manifest
    if AUTOGEN_OK and fixer_agent:
        try:
            fix_prompt = (
                f"Feature: {feature['desc']}\n"
                f"Given this pytest output:\n{test_output[-2000:]}\n\n"
                "Return ONLY a JSON manifest with a minimal set of file replacements to fix the errors."
            )
            res = await fixer_agent.on_messages(
                [{"role": "user", "content": fix_prompt}],
                cancellation_token=None
            )
            manifest = extract_manifest(res.messages[-1]["content"])
            fix_files = manifest.get("files", []) or []
        except Exception:
            fix_files = []
    # apply minimal patch if any
    applied = False
    if fix_files:
        try:
            base = PROJECT_ROOT / feature["id"]
            write_files(base, fix_files)
            applied = True
        except Exception as e:
            applied = False
    return applied, ("\n".join(steps) if steps else ""), fix_files

async def build_feature(feature: Dict[str, Any], proj: Dict[str, Any]):
    global CURRENT_FEATURE_ID
    CURRENT_FEATURE_ID = feature["id"]
    feature["status"] = "running"
    feature["progress"] = 5
    feature["started_at"] = now_iso()
    feature.setdefault("tries", 0)
    save_state(); write_static_dashboard()

    # GitHub issue for the feature (if missing)
    repo = (proj.get("github") or {}).get("repo") or GITHUB_REPO
    if repo and GITHUB_TOKEN and "issue" not in feature:
        issue_title = f"[{proj['name']}] {feature['id']}: {feature['desc']}"
        num = gh_create_issue(repo, issue_title, "Auto-tracked feature task.", ["feature","autogen"])
        if num:
            feature["issue"] = int(num)
            save_state()

    # Ask builder
    feature_dir = (PROJECT_ROOT / feature["id"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        f"Project: {proj['name']}\n"
        f"Feature: {feature['desc']}\n"
        "Return ONLY the JSON manifest as per your system message, including unit tests (pytest) where applicable."
    )

    try:
        if not AUTOGEN_OK or not OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="Model not available. Set OPENAI_API_KEY and install autogen packages.")

        res = await builder_agent.on_messages(
            [{"role": "user", "content": prompt}],
            cancellation_token=None
        )
        text = res.messages[-1]["content"]
        manifest = extract_manifest(text)
        files = manifest.get("files", [])
        if not files:
            raise HTTPException(status_code=400, detail="Manifest missing 'files'")
        written = write_files(feature_dir, files)
        feature["files"] = written
        feature["progress"] = 50
        save_state(); write_static_dashboard()
        if repo and feature.get("issue"):
            gh_comment_issue(repo, feature["issue"], f"Generated {len(written)} files.")
    except Exception as e:
        feature["status"] = "failed"; feature["progress"] = 100; feature["ended_at"] = now_iso()
        feature["error"] = f"Build error: {e}"
        feature["fix_steps"] = ["Open /control → check features", "Re-run Start", "If persists, check logs"]
        save_state(); write_static_dashboard(); checkpoint_to_github("build-error")
        if repo and feature.get("issue"):
            gh_update_issue_labels(repo, feature["issue"], ["autogen","build-error"])
            gh_comment_issue(repo, feature["issue"], f"Build error:\n\n```\n{e}\n```")
        CURRENT_FEATURE_ID = None
        return

    # Lint/format pass (best-effort)
    try:
        subprocess.run(["python", "-m", "pip", "install", "-q", "ruff", "black"], check=False)
        subprocess.run(["python", "-m", "ruff", "check", "--fix", "."], cwd=str(feature_dir), check=False)
        subprocess.run(["python", "-m", "black", "."], cwd=str(feature_dir), check=False)
    except Exception:
        pass

    # Tests
    passed, test_out = await run_pytests(feature_dir)
    (feature_dir / "_test_output.txt").write_text(test_out[-5000:], encoding="utf-8")
    if passed:
        feature["status"] = "done"; feature["progress"] = 100; feature["ended_at"] = now_iso()
        feature["error"] = ""; feature["fix_steps"] = []
        save_state(); write_static_dashboard(); checkpoint_to_github("tests-pass")
        if repo and feature.get("issue"):
            gh_update_issue_labels(repo, feature["issue"], ["autogen","done"])
            gh_comment_issue(repo, feature["issue"], "✅ Tests passed.")
        CURRENT_FEATURE_ID = None
        return

    # Auto-fix attempt
    feature["tries"] = int(feature.get("tries", 0)) + 1
    fixed = False
    steps_text = ""
    if feature["tries"] <= MAX_FIX_ATTEMPTS_PER_FEATURE:
        applied, steps_text, fix_files = await diagnose_and_fix(feature, proj, test_out)
        if steps_text:
            feature["fix_steps"] = [s for s in steps_text.splitlines() if s.strip()][:8]
        # retry tests if patch applied
        if applied:
            passed2, test_out2 = await run_pytests(feature_dir)
            (feature_dir / "_test_output_fix.txt").write_text(test_out2[-5000:], encoding="utf-8")
            if passed2:
                fixed = True
                feature["status"] = "done"; feature["progress"] = 100; feature["ended_at"] = now_iso()
                feature["error"] = ""; feature["fix_steps"] = feature.get("fix_steps", [])
                save_state(); write_static_dashboard(); checkpoint_to_github("autofix-pass")
                if repo and feature.get("issue"):
                    gh_update_issue_labels(repo, feature["issue"], ["autogen","done"])
                    gh_comment_issue(repo, feature["issue"], "✅ Auto-fix succeeded. Tests now pass.")
                CURRENT_FEATURE_ID = None
                return
            else:
                test_out = test_out2  # keep latest
    # Fail final
    if not fixed:
        feature["status"] = "failed"; feature["progress"] = 100; feature["ended_at"] = now_iso()
        feature["error"] = f"Tests failed.\n\n--- pytest tail ---\n{test_out[-2000:]}"
        if not feature.get("fix_steps"):
            feature["fix_steps"] = [
                "Open the feature folder in GitHub (checkpoint path) and review _test_output.txt",
                "Run tests locally to reproduce",
                "Apply minimal fixes to failing files",
                "Commit and re-run Start"
            ]
        save_state(); write_static_dashboard(); checkpoint_to_github("tests-failed")
        if repo and feature.get("issue"):
            gh_update_issue_labels(repo, feature["issue"], ["autogen","test-failure"])
            gh_comment_issue(repo, feature["issue"], f"❌ Tests failed. Tail:\n\n```\n{test_out[-1000:]}\n```")
    CURRENT_FEATURE_ID = None

# =====================================================================================
# Runner
# =====================================================================================

def overall_ordered_features() -> List[Dict[str, Any]]:
    proj = STATE.get("project")
    if not proj or not proj.get("features"):
        return []
    by_id = {f["id"]: f for f in proj["features"]}
    order = proj.get("order") or []
    return [by_id[i] for i in order if i in by_id] + [f for f in proj["features"] if f["id"] not in order]

async def runner_loop():
    global RUNNING
    RUNNING = True
    while RUNNING:
        proj = STATE.get("project")
        if proj:
            features = overall_ordered_features()
            running_exists = any(f.get("status") == "running" for f in features)
            queued = next((f for f in features if f.get("status") == "queued"), None)
            if (not running_exists) and queued:
                await build_feature(queued, proj)
            else:
                # nothing queued — still checkpoint regularly so next server can resume seamlessly
                checkpoint_to_github("heartbeat")
        await asyncio.sleep(2.0)

# =====================================================================================
# Startup auto-restore (optional)
# =====================================================================================

@app.on_event("startup")
async def on_startup():
    # attempt restore from GitHub if configured & project exists locally
    try:
        if GH_AUTORESTORE:
            restored = restore_from_github()
            if restored:
                write_static_dashboard()
    except Exception:
        pass

# =====================================================================================
# Views & Control
# =====================================================================================

@app.get("/", response_class=HTMLResponse)
def static_dashboard():
    try:
        idx = SITE_DIR / "index.html"
        if not idx.exists():
            write_static_dashboard()
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    except Exception as e:
        return HTMLResponse(f"<pre>Dashboard not ready.\n{e}</pre>", status_code=200)

@app.get("/control", response_class=HTMLResponse)
def control():
    proj = STATE.get("project") or {}
    name = proj.get("name", "")
    desc = proj.get("description", "")
    gh = proj.get("github") or {}
    repo = gh.get("repo", GITHUB_REPO or "")
    branch = gh.get("branch", GITHUB_BRANCH)
    dirpref = gh.get("dir", GITHUB_DIR)
    running = "running" if RUNNING else "stopped"
    return f"""<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Head of the Snake · Control</title>
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
<h2>Head of the Snake — Control Panel</h2>
<p><a href="/">View Static Dashboard</a> · <small>Runner: <b>{running}</b></small></p>

<h3>1) Define / Update Single Project</h3>
<form onsubmit="return false">
<label>Project name</label>
<input id="pname" value="{name}" placeholder="e.g., research-agent">
<label>Description</label>
<textarea id="pdesc" placeholder="Short description">{desc}</textarea>
<label>GitHub repo (owner/repo, for checkpoint + issues)</label>
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
<button onclick="checkpoint()">Checkpoint Now</button>
<button onclick="restore()">Restore From GitHub</button>
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
async function checkpoint(){{
  const r = await fetch('/checkpoint', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}
async function restore(){{
  const r = await fetch('/restore', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}
async function regen(){{
  const r = await fetch('/regen', {{method:'POST'}});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}
</script>
</body>
</html>"""

# =====================================================================================
# API (defensive)
# =====================================================================================

@app.get("/health")
def health():
    proj_name = (STATE.get("project") or {}).get("name")
    return {"status": "ok", "project": proj_name, "running": RUNNING, "workspace": str(WORKSPACE)}

@app.post("/project")
def set_project(body: Dict[str, Any] = Body(...)):
    if not AUTOGEN_OK:
        raise HTTPException(status_code=400, detail="Model packages missing. Add autogen-agentchat and autogen-ext[openai].")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set.")

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
            "dir": (gh.get("dir") or GITHUB_DIR),
        },
        "features": [],
        "order": [],
    }
    save_state(); write_static_dashboard(); checkpoint_to_github("project-set")
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
            "error": None,
            "fix_steps": [],
            "tries": 0
        })
    proj["features"] = feats
    proj["order"] = [x["id"] for x in feats]
    save_state(); write_static_dashboard(); checkpoint_to_github("features-set")
    return {"ok": True, "count": len(feats)}

@app.post("/plan")
async def plan_features():
    if not AUTOGEN_OK or not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="Model not available. Set OPENAI_API_KEY and install autogen packages.")
    proj = STATE.get("project")
    if not proj:
        raise HTTPException(status_code=400, detail="Set project first")
    if not proj.get("description"):
        raise HTTPException(status_code=400, detail="Project description required to plan")

    res = await planner_agent.on_messages(
        [{"role": "user", "content": f"Project: {proj['name']}\nDescription: {proj['description']}\nList features:"}],
        cancellation_token=None
    )
    text = (res.messages[-1]["content"] if res and res.messages else "").strip()
    candidates = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    if not candidates:
        raise HTTPException(status_code=400, detail="Planner returned no features")
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
            "error": None,
            "fix_steps": [],
            "tries": 0
        })
    proj["features"] = feats
    proj["order"] = [x["id"] for x in feats]
    save_state(); write_static_dashboard(); checkpoint_to_github("features-planned")
    return {"ok": True, "planned": len(feats)}

@app.post("/prioritize")
async def prioritize():
    if not AUTOGEN_OK or not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="Model not available. Set OPENAI_API_KEY and install autogen packages.")
    proj = STATE.get("project")
    if not proj or not proj.get("features"):
        raise HTTPException(status_code=400, detail="No features to prioritize")
    lines = "\n".join([f"- {f['desc']}" for f in proj["features"]])
    res = await prioritizer_agent.on_messages(
        [{"role": "user", "content": f"Features:\n{lines}\nReorder top-to-bottom for fastest MVP and most functionality."}],
        cancellation_token=None
    )
    text = (res.messages[-1]["content"] if res and res.messages else "").strip()
    ordered_desc = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    desc_to_id = {f["desc"]: f["id"] for f in proj["features"]}
    new_order = []
    for d in ordered_desc:
        if d in desc_to_id:
            new_order.append(desc_to_id[d])
    for f in proj["features"]:
        if f["id"] not in new_order:
            new_order.append(f["id"])
    proj["order"] = new_order
    save_state(); write_static_dashboard(); checkpoint_to_github("prioritized")
    return {"ok": True, "order_count": len(new_order)}

@app.post("/runner/start")
async def start_runner():
    global RUNNING
    if not AUTOGEN_OK or not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="Model not available. Set OPENAI_API_KEY and install autogen packages.")
    if not RUNNING:
        asyncio.create_task(runner_loop())
    write_static_dashboard(); checkpoint_to_github("runner-start")
    return {"running": True}

@app.post("/runner/stop")
async def stop_runner():
    global RUNNING
    RUNNING = False
    write_static_dashboard(); checkpoint_to_github("runner-stop")
    return {"running": False}

@app.post("/checkpoint")
def checkpoint():
    checkpoint_to_github("manual")
    return {"ok": True}

@app.post("/restore")
def restore():
    ok = restore_from_github()
    if ok:
        write_static_dashboard()
    return {"ok": bool(ok)}

@app.post("/regen")
def regen():
    write_static_dashboard()
    return {"ok": True}

@app.get("/status")
def status():
    proj = STATE.get("project")
    return {
        "running": RUNNING,
        "current_feature": CURRENT_FEATURE_ID,
        "progress_percent": overall_progress(),
        "project": proj or None
    }

@app.get("/download")
def download(feature_id: Optional[str] = Query(None)):
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        return PlainTextResponse(f"Download error: {e}", status_code=400)
