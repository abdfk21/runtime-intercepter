"""Synthesizer Agent: Merges multi-agent outputs into final answer."""

from app.agents.base_agent import BaseAgent
from app.llm.base import BaseLLMProvider


class SynthesizerAgent(BaseAgent):
    def __init__(self, name: str = "synthesizer", llm_provider: BaseLLMProvider = None):
        system_prompt = (
            "You are the Synthesizer Agent in a multi-agent system. "
            "Your objective is to produce the final comprehensive, coherent, and verified answer. "
            "Synthesize all verified claims from upstream stages, resolve remaining nuances, "
            "and produce an observable, clear, and well-structured final output."
        )
        super().__init__(
            name=name,
            role="Consensus Synthesis and Final Output Generation",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
        )
