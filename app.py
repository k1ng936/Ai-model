# app.py
# Chat -> GPT -> agent instructions -> optional email dispatch
# Run: uvicorn app:app --host 0.0.0.0 --port 8080

import os
import json
import smtplib
from email.message import EmailMessage
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ValidationError
from dotenv import load_dotenv

# --- Load env
load_dotenv()

# === ENV ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # set your default model
SYSTEM_NAME = os.getenv("SYSTEM_NAME", "Team Churchill Controller")

# Agents as a comma-separated list of emails
# Example: "agent1@your.com,agent2@your.com,agent3@your.com"
AGENT_EMAILS_CSV = os.getenv("AGENT_EMAILS", "")
DEFAULT_AGENT_EMAILS = [e.strip() for e in AGENT_EMAILS_CSV.split(",") if e.strip()]

# SMTP (use your provider; for Gmail consider an app password)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME or "no-reply@localhost")

# Safety toggles
ALLOW_EMAIL_SEND = os.getenv("ALLOW_EMAIL_SEND", "true").lower() == "true"

# --- OpenAI: use the official SDK
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Install openai: pip install openai>=1.0") from e

client = OpenAI(api_key=OPENAI_API_KEY)

# === FastAPI app ===
app = FastAPI(title=SYSTEM_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic models ===
class AgentTask(BaseModel):
    to: EmailStr = Field(..., description="Agent email recipient")
    subject: str = Field(..., max_length=200)
    body: str = Field(..., description="Plaintext instruction body")

class OrchestratorOutput(BaseModel):
    assistant_reply: str = Field(..., description="What the assistant says back to the user")
    tasks: List[AgentTask] = Field(default_factory=list, description="Emails to send to agents")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    dry_run: bool = Field(default=False, description="If true, do NOT send emails; just preview tasks.")
    agent_emails: Optional[List[EmailStr]] = Field(default=None, description="Override default agent list")

class ChatResponse(BaseModel):
    assistant_reply: str
    dispatched_emails: List[Dict[str, str]] = Field(default_factory=list)
    preview_tasks: Optional[List[AgentTask]] = None


# === System prompt for the model ===
SYSTEM_PROMPT = f"""
You are {SYSTEM_NAME}, a controller that converts the user's plain-English request into:
1) A concise chat reply for the user.
2) A set of actionable email tasks for human/agent inboxes.

OUTPUT FORMAT (MUST be strict JSON):
{{
  "assistant_reply": "string",
  "tasks": [
    {{"to":"email","subject":"string","body":"string"}}
  ]
}}

RULES:
- Only create tasks that are concrete and immediately doable by agents.
- Default recipients are the provided agent emails; assign tasks to specific recipients if user implies roles.
- Be explicit, numbered, and time-bounded in bodies (include deadlines, deliverables, links, and success criteria).
- Keep subjects short and action-focused (e.g., "Draft outreach list for Toronto DJs by EOD").
- If no agent work is needed, return an empty tasks array.
"""

# === Helpers ===
def call_model(user_message: str, agent_emails: List[str]) -> OrchestratorOutput:
    # Instruct the model about who is available
    availability_note = (
        "Available agent inboxes:\n- " + "\n- ".join(agent_emails) if agent_emails else "No agent emails provided."
    )
    user_block = f"{availability_note}\n\nUSER MESSAGE:\n{user_message}\n\nRemember: return STRICT JSON only."

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()

    # Try to parse strict JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: ask the model to fix JSON (one quick repair step)
        fix = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Fix this into strict JSON. Do not add commentary."},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
        )
        data = json.loads(fix.choices[0].message.content.strip())

    try:
        return OrchestratorOutput(**data)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Model output validation error: {e}")


def send_email(task: AgentTask) -> None:
    if not (SMTP_HOST and SMTP_PORT and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM):
        raise RuntimeError("SMTP not configured. Set SMTP_HOST/PORT/USERNAME/PASSWORD/SMTP_FROM.")

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = task.to
    msg["Subject"] = task.subject
    msg.set_content(task.body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.send_message(msg)


def dispatch_tasks(tasks: List[AgentTask]) -> List[Dict[str, str]]:
    dispatched = []
    for t in tasks:
        send_email(t)
        dispatched.append({"to": t.to, "subject": t.subject})
    return dispatched


# === Routes ===
@app.get("/", summary="Health check")
def root():
    return {"ok": True, "service": SYSTEM_NAME}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    agent_emails = req.agent_emails or DEFAULT_AGENT_EMAILS
    out = call_model(req.message, agent_emails)

    # If the model didn't fill recipients, fan tasks out in round-robin
    prepared_tasks: List[AgentTask] = []
    if out.tasks:
        if agent_emails:
            # Fill missing 'to' fields sensibly
            rr_i = 0
            for t in out.tasks:
                if not t.to:
                    t.to = agent_emails[rr_i % len(agent_emails)]
                    rr_i += 1
                prepared_tasks.append(t)
        else:
            prepared_tasks = out.tasks  # as-is

    if req.dry_run or not ALLOW_EMAIL_SEND:
        return ChatResponse(
            assistant_reply=out.assistant_reply,
            dispatched_emails=[],
            preview_tasks=prepared_tasks or out.tasks,
        )

    # Send emails
    dispatched = []
    for task in prepared_tasks:
        try:
            send_email(task)
            dispatched.append({"to": task.to, "subject": task.subject})
        except Exception as e:
            # Don't fail the whole request if one email fails; report partials
            dispatched.append({"to": task.to, "subject": task.subject, "error": str(e)})

    return ChatResponse(
        assistant_reply=out.assistant_reply,
        dispatched_emails=dispatched,
        preview_tasks=None,
    )
