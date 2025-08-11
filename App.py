from fastapi import FastAPI
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

app = FastAPI()

# Uses OPENAI_API_KEY from environment (set in Koyeb)
model = OpenAIChatCompletionClient(model="gpt-4o-mini")

@app.get("/")
def health():
    return {"status": "ok", "service": "autogen-demo"}

@app.get("/demo")
async def demo():
    agent = AssistantAgent("helper", model_client=model)
    result = await agent.on_messages([{"role": "user", "content": "Say hello in one short sentence."}])
    return {"reply": result.messages[-1]["content"]}
