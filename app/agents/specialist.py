"""Specialist Agent: Provides domain-specific technical deep-dives."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class SpecialistAgent(BaseAgent):
    def __init__(self, name: str = "specialist", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Specialist Agent in a hierarchical multi-agent system. "
            "You provide domain-specific knowledge, detailed technical or mathematical rigor, "
            "and ensure precise operational definitions and empirical support."
        )
        super().__init__(
            name=name,
            role="Domain Specialist and Deep Technical Analysis",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
