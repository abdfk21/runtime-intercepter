"""Critic Agent: Evaluates counterarguments, ambiguities, and edge cases."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class CriticAgent(BaseAgent):
    def __init__(self, name: str = "critic", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Critic Agent in a multi-agent system. "
            "Your role is to detect internal inconsistencies, unwarranted assumptions, overconfidence, "
            "and potential edge cases or counter-evidence that previous agents may have overlooked."
        )
        super().__init__(
            name=name,
            role="Critical Analysis and Anomaly Detection",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
