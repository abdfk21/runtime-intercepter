"""Planner Agent: Formulates task decomposition and reasoning paths."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class PlannerAgent(BaseAgent):
    def __init__(self, name: str = "planner", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Planner Agent. "
            "Your objective is to decompose complex research questions into structured subtasks, "
            "identify essential dependencies, and define verification criteria."
        )
        super().__init__(
            name=name,
            role="Task Decomposition and Strategic Planning",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
