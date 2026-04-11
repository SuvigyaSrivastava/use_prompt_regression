import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

class PromptExecutor:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    def run(self, prompt: str, user_input: str = "Run the task.") -> str:
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[executor error] {e}")
            return ""

    def run_batch(self, prompt: str, inputs: list) -> list:
        return [self.run(prompt, inp) for inp in inputs]