"""Verifier Agent: Cross-checks claims against evidence and context."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class VerifierAgent(BaseAgent):
    def __init__(self, name: str = "verifier", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Verifier Agent in a multi-agent system. "
            "Your role is to rigorously cross-check the claims provided by upstream agents. "
            "Verify whether each claim is supported by the cited evidence and background context. "
            "Flag any unsupported claims, numerical discrepancies, or logical non-sequiturs."
        )
        super().__init__(
            name=name,
            role="Factual Verification and Evidence Cross-Checking",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
