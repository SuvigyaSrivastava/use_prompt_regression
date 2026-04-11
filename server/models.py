from pydantic import BaseModel
from typing import List

try:
    from openenv_core.models import BaseAction, BaseObservation, BaseState
except ImportError:
    class BaseAction(BaseModel): pass
    class BaseObservation(BaseModel): pass
    class BaseState(BaseModel): pass

class AssertionResult(BaseModel):
    assertion_id: str
    description: str
    passed: bool
    score: float

class PromptAction(BaseAction):
    prompt: str

class PromptObservation(BaseObservation):
    task_id: str
    task_description: str
    broken_prompt: str
    current_prompt: str
    llm_output: str
    assertion_results: List[AssertionResult]
    tests_passed: int
    tests_total: int
    reward: float
    step_number: int
    done: bool
    info: str

class PromptState(BaseState):
    task_id: str
    task_description: str
    broken_prompt: str
    current_prompt: str
    llm_output: str
    assertion_results: List[AssertionResult]
    tests_passed: int
    tests_total: int
    total_reward: float
    step_number: int
    max_steps: int
    done: bool
    info: str
    prompt_history: List[str]