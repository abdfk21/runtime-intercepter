"""Researcher Agent: Gathers initial claims and evidence."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class ResearcherAgent(BaseAgent):
    def __init__(self, name: str = "researcher", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Researcher Agent in an academic multi-agent research pipeline. "
            "Your objective is to thoroughly investigate the given topic or question, extract atomic factual claims, "
            "provide explicit observable evidence for each claim, and declare confidence levels accurately. "
            "Do not state claims without grounded evidence. If uncertain, state a lower confidence."
        )
        super().__init__(
            name=name,
            role="Information Gathering and Initial Evidence Extraction",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
