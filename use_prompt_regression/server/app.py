import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from models import PromptAction, PromptObservation, PromptState
from environment import PromptRegressionEnv

app = FastAPI(title="use_prompt_regression")
env = PromptRegressionEnv()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
async def reset(body: dict = {}):
    task_id = body.get("task_id", None) if body else None
    result = await env.reset(task_id=task_id)
    return {
        "observation": result.observation.dict(),
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
async def step(action: PromptAction):
    result = await env.step(action)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

@app.get("/state")
async def state():
    s = await env.state()
    return s.dict()