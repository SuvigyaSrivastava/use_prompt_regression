import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

try:
    from models import PromptAction, PromptObservation, PromptState, AssertionResult
    from tasks import TaskBank
    from executor import PromptExecutor
except ModuleNotFoundError:
    from server.models import PromptAction, PromptObservation, PromptState, AssertionResult
    from server.tasks import TaskBank
    from server.executor import PromptExecutor

@dataclass
class ResetResult:
    observation: PromptObservation

@dataclass
class StepResult:
    observation: PromptObservation
    reward: float
    done: bool
    info: dict

class PromptRegressionEnv:
    def __init__(self):
        self.task_bank = TaskBank()
        self.executor = PromptExecutor()
        self._task_cycle = [
            "task_json_formatter",
            "task_sentiment_classifier",
            "task_adversarial_follower"
        ]
        self._cycle_index = 0
        self._reset_state()

    def _reset_state(self):
        self.current_task_id = None
        self.current_prompt = ""
        self.llm_output = ""
        self.assertion_results = []
        self.step_number = 0
        self.max_steps = 5
        self.done = False
        self.prompt_history = []
        self.total_reward = 0.0
        self.last_reward = 0.0

    async def reset(self, task_id: str = None) -> ResetResult:
        if task_id is None:
            task_id = self._task_cycle[self._cycle_index % len(self._task_cycle)]
            self._cycle_index += 1
        self._reset_state()
        task = self.task_bank.get_task(task_id)
        self.current_task_id = task_id
        self.current_prompt = task.broken_prompt
        self.max_steps = task.max_steps
        if task.broken_prompt:
            if task_id == "task_json_formatter":
                self.llm_output = self.executor.run(task.broken_prompt)
                self.assertion_results, self.last_reward = self.task_bank.run_assertions(task_id, self.llm_output)
            else:
                outputs = self.executor.run_batch(task.broken_prompt, task.test_inputs)
                all_results = []
                scores = []
                for inp, out in zip(task.test_inputs, outputs):
                    res, sc = self.task_bank.run_assertions(task_id, out, test_input=inp, single_input=True)
                    all_results.extend(res)
                    scores.append(sc)
                self.assertion_results = all_results
                self.last_reward = sum(scores) / len(scores) if scores else 0.0
                self.llm_output = outputs[0] if outputs else ""
        else:
            self.llm_output = ""
            self.assertion_results = []
            self.last_reward = 0.0
        obs = self._build_observation(task)
        return ResetResult(observation=obs)

    async def step(self, action: PromptAction) -> StepResult:
        self.step_number += 1
        task = self.task_bank.get_task(self.current_task_id)
        new_prompt = action.prompt[:2000]
        if new_prompt in self.prompt_history:
            self.last_reward = 0.0
            obs = self._build_observation(task)
            return StepResult(observation=obs, reward=0.0, done=False, info={"msg": "Duplicate prompt penalised"})
        self.current_prompt = new_prompt
        self.prompt_history.append(new_prompt)
        if self.current_task_id == "task_json_formatter":
            self.llm_output = self.executor.run(new_prompt)
            self.assertion_results, base_reward = self.task_bank.run_assertions(self.current_task_id, self.llm_output)
        else:
            outputs = self.executor.run_batch(new_prompt, task.test_inputs)
            all_results = []
            scores = []
            for inp, out in zip(task.test_inputs, outputs):
                res, sc = self.task_bank.run_assertions(self.current_task_id, out, test_input=inp, single_input=True)
                all_results.extend(res)
                scores.append(sc)
            self.assertion_results = all_results
            base_reward = sum(scores) / len(scores) if scores else 0.0
            self.llm_output = outputs[0] if outputs else ""
        if len(new_prompt) > 1500:
            base_reward *= 0.9
        self.last_reward = round(min(max(base_reward, 0.0), 1.0), 4)
        self.total_reward += self.last_reward
        tests_passed = sum(1 for r in self.assertion_results if r.passed)
        tests_total = len(self.assertion_results)
        self.done = (self.last_reward >= 0.95) or (self.step_number >= self.max_steps)
        info_msg = f"Step {self.step_number}: {tests_passed}/{tests_total} passed. {'Solved!' if self.done and self.last_reward >= 0.95 else 'Keep refining.'}"
        obs = self._build_observation(task)
        return StepResult(observation=obs, reward=self.last_reward, done=self.done, info={"msg": info_msg})

    async def state(self) -> PromptState:
        task = self.task_bank.get_task(self.current_task_id) if self.current_task_id else None
        tests_passed = sum(1 for r in self.assertion_results if r.passed)
        return PromptState(
            task_id=self.current_task_id or "",
            task_description=task.description if task else "",
            broken_prompt=task.broken_prompt if task else "",
            current_prompt=self.current_prompt,
            llm_output=self.llm_output,
            assertion_results=self.assertion_results,
            tests_passed=tests_passed,
            tests_total=len(self.assertion_results),
            total_reward=self.total_reward,
            step_number=self.step_number,
            max_steps=self.max_steps,
            done=self.done,
            info="",
            prompt_history=self.prompt_history
        )

    def _build_observation(self, task) -> PromptObservation:
        tests_passed = sum(1 for r in self.assertion_results if r.passed)
        return PromptObservation(
            task_id=self.current_task_id,
            task_description=task.description,
            broken_prompt=task.broken_prompt,
            current_prompt=self.current_prompt,
            llm_output=self.llm_output,
            assertion_results=self.assertion_results,
            tests_passed=tests_passed,
            tests_total=len(self.assertion_results),
            reward=self.last_reward,
            step_number=self.step_number,
            done=self.done,
            info=f"Step {self.step_number}: {tests_passed}/{len(self.assertion_results)} passed"
        )