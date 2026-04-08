import asyncio
import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'use_prompt_regression', 'server'))

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "use_prompt_regression")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")
MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.8
TASKS = ["task_json_formatter", "task_sentiment_classifier", "task_adversarial_follower"]

def log_start(task, env, model):
    print(json.dumps({"type": "START", "task": task, "env": env, "model": model}), flush=True)

def log_step(step, action, reward, done, error):
    print(json.dumps({"type": "STEP", "step": step, "action": action[:200], "reward": reward, "done": done, "error": error}), flush=True)

def log_end(success, steps, score, rewards):
    print(json.dumps({"type": "END", "success": success, "steps": steps, "score": score, "rewards": rewards}), flush=True)

def get_agent_prompt(client, obs, history):
    system = """You are an expert prompt engineer.
Your job is to rewrite a broken prompt so that an LLM produces outputs passing all test assertions.
Study failed assertions carefully. Write a precise, instruction-following system prompt.
Output your reasoning first, then wrap your final prompt in <prompt> and </prompt> tags.
Keep the prompt under 1000 characters."""

    failed = [a for a in obs.assertion_results if not a.passed]
    user = f"""Task: {obs.task_description}
Original broken prompt: {obs.broken_prompt}
Your last prompt: {obs.current_prompt}
LLM output: {obs.llm_output[:300]}
Failed assertions: {json.dumps([{{'id': a.assertion_id, 'desc': a.description}} for a in failed], indent=2)}
Steps taken: {obs.step_number}
Previous prompts tried: {history[-3:] if history else 'none'}

Write a better prompt that fixes ALL failing assertions."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=600,
            timeout=30
        )
        text = resp.choices[0].message.content
        if "<prompt>" in text and "</prompt>" in text:
            return text.split("<prompt>")[1].split("</prompt>")[0].strip()
        return text.strip()[:1000]
    except Exception as e:
        print(f"[agent error] {e}", flush=True)
        return obs.broken_prompt + " Respond concisely and follow the format exactly."

async def run_task(task_id: str) -> float:
    from environment import PromptRegressionEnv
    from models import PromptAction

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = PromptRegressionEnv()
    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=IMAGE_NAME, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            new_prompt = get_agent_prompt(client, obs, history)
            result = await env.step(PromptAction(prompt=new_prompt))
            obs = result.observation
            reward = result.reward
            done = result.done
            rewards.append(reward)
            steps_taken = step
            history.append(new_prompt)
            log_step(step=step, action=new_prompt, reward=reward, done=done, error=None)
            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken, action="", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

async def main():
    scores = []
    for task_id in TASKS:
        score = await run_task(task_id)
        scores.append(score)
    print(json.dumps({
        "type": "SUMMARY",
        "task_scores": dict(zip(TASKS, scores)),
        "mean_score": round(sum(scores) / len(scores), 4)
    }), flush=True)

if __name__ == "__main__":
    asyncio.run(main())