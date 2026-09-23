"""Supervisor Agent: Coordinates sub-agents in Hierarchical architecture."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class SupervisorAgent(BaseAgent):
    def __init__(self, name: str = "supervisor", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Supervisor Agent managing a hierarchical multi-agent team. "
            "You formulate sub-directives for specialist agents (researcher, verifier, specialist, critic), "
            "evaluate incoming reports, maintain global context, and prepare consolidated context for synthesis."
        )
        super().__init__(
            name=name,
            role="Hierarchical Team Coordinator and Subtask Dispatcher",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
