import os, re, json, base64, pathlib, time
from typing import Dict, Any, List
from fastapi import FastAPI, Query, Body, HTTPException
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import requests

app = FastAPI(title="AutoGen Builder")

# ====== Config ======
MODEL_NAME = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
WORKSPACE_ROOT = pathlib.Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# GitHub (optional)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
TARGET_REPO  = os.getenv("TARGET_REPO", "")  # e.g. "yourname/yourrepo"
TARGET_BRANCH = os.getenv("TARGET_BRANCH", "main")
TARGET_DIR = os.getenv("TARGET_DIR", "autogen-output")  # subfolder in repo

# ====== AutoGen Agent ======
model = OpenAIChatCompletionClient(model=MODEL_NAME)
assistant = AssistantAgent(
    "builder",
    model_client=model,
    system_message=(
        "You are a senior software engineer. "
        "When asked to 'build' an app, respond ONLY with a single JSON object in this schema:\n"
        "{\n"
        '  "files": [{"path": "relative/path.ext", "content": "file content as UTF-8 text"}],\n'
        '  "instructions": "how to run",\n'
        '  "postbuild": ["optional shell commands"]\n'
        "}\n"
        "Do not include commentary. If the request is unclear, choose sensible defaults and still return valid JSON. "
        "Keep paths POSIX-style. Avoid huge binaries; use placeholders where needed."
    ),
)

# ====== Helpers ======
JSON_BLOCK = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def extract_json_manifest(text: str) -> Dict[str, Any]:
    """Find a JSON code block or last JSON-looking object."""
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
    # Get existing SHA if file exists
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

# ====== Endpoints ======
@app.get("/")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE_ROOT)}

@app.get("/agent")
async def agent(task: str = Query(..., description="What you want AutoGen to do")):
    res = await assistant.on_messages([{"role": "user", "content": task}])
    return {"task": task, "reply": res.messages[-1]["content"]}

@app.post("/build")
async def build(
    body: Dict[str, Any] = Body(
        ..., example={"goal": "Build a simple TODO web app with FastAPI and SQLite"}
    )
):
    goal = body.get("goal")
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' field.")
    session_id = f"proj-{int(time.time())}"
    base = (WORKSPACE_ROOT / session_id)
    base.mkdir(parents=True, exist_ok=True)

    # Ask AutoGen to produce the manifest JSON
    prompt = (
        f"Goal: {goal}\n"
        "Output only the JSON manifest as described in your system message. "
        "Keep it reasonably small and runnable. Include a README.md in files."
    )
    res = await assistant.on_messages([{"role": "user", "content": prompt}])
    text = res.messages[-1]["content"]

    manifest = extract_json_manifest(text)
    files = manifest.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="No 'files' in manifest.")

    written = write_files(base, files)

    # Optionally push to GitHub
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
