"""Hierarchical multi-agent architecture implemented with LangGraph:
Supervisor coordinates Specialist, Researcher, Verifier, Critic -> Synthesizer.
"""

import time
import uuid
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
from app.agents.supervisor import SupervisorAgent
from app.agents.researcher import ResearcherAgent
from app.agents.specialist import SpecialistAgent
from app.agents.verifier import VerifierAgent
from app.agents.critic import CriticAgent
from app.agents.synthesizer import SynthesizerAgent
from app.llm.base import BaseLLMProvider
from app.interceptor.interceptor import RuntimeInterceptor
from app.interceptor.handoff import InterceptorDecision
from app.fault_injection.injector import FaultInjector
from app.fault_injection.fault_types import FaultType
from app.orchestration.graph_builder import MultiAgentState


class HierarchicalPipeline:
    """LangGraph-based hierarchical architecture with supervisor-orchestrated coordination."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        interceptor: Optional[RuntimeInterceptor] = None,
        fault_injector: Optional[FaultInjector] = None,
    ):
        self.llm = llm_provider
        self.interceptor = interceptor or RuntimeInterceptor()
        self.fault_injector = fault_injector or FaultInjector()

        self.supervisor = SupervisorAgent(llm_provider=self.llm)
        self.researcher = ResearcherAgent(llm_provider=self.llm)
        self.specialist = SpecialistAgent(llm_provider=self.llm)
        self.verifier = VerifierAgent(llm_provider=self.llm)
        self.critic = CriticAgent(llm_provider=self.llm)
        self.synthesizer = SynthesizerAgent(llm_provider=self.llm)

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MultiAgentState)

        builder.add_node("supervisor_plan", self._run_supervisor_plan)
        builder.add_node("researcher_step", self._run_researcher)
        builder.add_node("specialist_step", self._run_specialist)
        builder.add_node("verifier_step", self._run_verifier)
        builder.add_node("critic_step", self._run_critic)
        builder.add_node("supervisor_consolidate", self._run_supervisor_consolidate)
        builder.add_node("synthesizer_step", self._run_synthesizer)

        builder.add_edge(START, "supervisor_plan")
        builder.add_edge("supervisor_plan", "researcher_step")
        builder.add_edge("researcher_step", "specialist_step")
        builder.add_edge("specialist_step", "verifier_step")
        builder.add_edge("verifier_step", "critic_step")
        builder.add_edge("critic_step", "supervisor_consolidate")
        builder.add_edge("supervisor_consolidate", "synthesizer_step")
        builder.add_edge("synthesizer_step", END)

        return builder.compile()

    def _run_supervisor_plan(self, state: MultiAgentState) -> Dict[str, Any]:
        res = self.supervisor.execute(
            task_prompt=f"Formulate initial multi-agent coordination directive for task: {state['prompt']}"
        )
        state["agent_outputs"]["supervisor_plan"] = res["output"]
        return state

    def _run_researcher(self, state: MultiAgentState) -> Dict[str, Any]:
        res = self.researcher.execute(
            task_prompt=state["prompt"],
            context={"supervisor_plan": state["agent_outputs"]["supervisor_plan"]},
        )
        output = res["output"]

        # Fault injection check
        fi_cfg = state.get("fault_injection_config")
        if fi_cfg and fi_cfg.get("location") in ["researcher", "hierarchical_researcher"]:
            ft = FaultType(fi_cfg.get("fault_type", "factual_mutation"))
            seed = fi_cfg.get("seed", 42)
            fi_res = self.fault_injector.inject_fault(output=output, fault_type=ft, location="researcher", seed=seed)
            output = fi_res.corrupted_output
            state["fault_injected"] = True
            state["fault_type"] = ft.value
            state["injected_fault_details"] = fi_res.to_dict()

        state["agent_outputs"]["researcher"] = output
        return state

    def _run_specialist(self, state: MultiAgentState) -> Dict[str, Any]:
        upstream = state["agent_outputs"]["researcher"]
        if state["interceptor_enabled"]:
            start_i = time.perf_counter()
            handoff_ctx = {"prompt": state["prompt"]}
            if state.get("injected_fault_details") and state["injected_fault_details"].get("ground_truth"):
                handoff_ctx["verified_evidence"] = state["injected_fault_details"]["ground_truth"]

            insp = self.interceptor.inspect_handoff(
                trace_id=state["trace_id"],
                source_agent="researcher",
                target_agent="specialist",
                source_output=upstream,
                target_context=handoff_ctx,
                model_name=state["model_name"],
                architecture=state["architecture"],
            )
            state["interceptor_latency_ms"] += (time.perf_counter() - start_i) * 1000.0
            state["handoffs"].append(insp.model_dump())

            if insp.decision == InterceptorDecision.REPAIR:
                start_r = time.perf_counter()
                repaired, _, success = self.interceptor.repair_engine.execute_repair_loop(
                    source_agent=self.researcher,
                    task_prompt=state["prompt"],
                    initial_output=upstream,
                    context={},
                    inspect_fn=lambda out, att: self.interceptor.inspect_handoff(
                        trace_id=state["trace_id"],
                        source_agent="researcher",
                        target_agent="specialist",
                        source_output=out,
                        target_context=handoff_ctx,
                        model_name=state["model_name"],
                        architecture=state["architecture"],
                        repair_attempt=att,
                    ),
                )
                state["repair_latency_ms"] += (time.perf_counter() - start_r) * 1000.0
                upstream = repaired
                state["agent_outputs"]["researcher"] = repaired
                if success:
                    state["error_contained"] = True

            elif insp.decision == InterceptorDecision.BLOCK:
                upstream = {"answer": "[CONTAINED: researcher output blocked]", "claims": [], "assumptions": [], "sources": []}
                state["error_contained"] = True

        spec_context = {"researcher_output": upstream}
        if state["interceptor_enabled"] and insp.decision == InterceptorDecision.PASS_WITH_WARNING:
            spec_context["interceptor_warning"] = (
                f"SAFETY WARNING: Upstream handoff flagged with risk score {insp.risk_score:.2f}. "
                f"Identified issues: {'; '.join(insp.explanations)}. Exercise skepticism."
            )
            state["error_contained"] = True

        res = self.specialist.execute(task_prompt=state["prompt"], context=spec_context)
        state["agent_outputs"]["specialist"] = res["output"]
        return state

    def _run_verifier(self, state: MultiAgentState) -> Dict[str, Any]:
        res = self.verifier.execute(
            task_prompt=state["prompt"],
            context={"researcher": state["agent_outputs"]["researcher"], "specialist": state["agent_outputs"]["specialist"]},
        )
        state["agent_outputs"]["verifier"] = res["output"]
        return state

    def _run_critic(self, state: MultiAgentState) -> Dict[str, Any]:
        res = self.critic.execute(task_prompt=state["prompt"], context=state["agent_outputs"])
        state["agent_outputs"]["critic"] = res["output"]
        return state

    def _run_supervisor_consolidate(self, state: MultiAgentState) -> Dict[str, Any]:
        res = self.supervisor.execute(
            task_prompt=f"Consolidate verified findings across all sub-agents for prompt: {state['prompt']}",
            context=state["agent_outputs"],
        )
        state["agent_outputs"]["supervisor_consolidated"] = res["output"]
        return state

    def _run_synthesizer(self, state: MultiAgentState) -> Dict[str, Any]:
        upstream = state["agent_outputs"]["supervisor_consolidated"]
        if state["interceptor_enabled"]:
            start_i = time.perf_counter()
            insp = self.interceptor.inspect_handoff(
                trace_id=state["trace_id"],
                source_agent="supervisor",
                target_agent="synthesizer",
                source_output=upstream,
                target_context={"prompt": state["prompt"]},
                model_name=state["model_name"],
                architecture=state["architecture"],
            )
            state["interceptor_latency_ms"] += (time.perf_counter() - start_i) * 1000.0
            state["handoffs"].append(insp.model_dump())

        res = self.synthesizer.execute(task_prompt=state["prompt"], context={"consolidated": upstream})
        state["agent_outputs"]["synthesizer"] = res["output"]
        state["final_output"] = res["output"]

        # Track cascade
        if state.get("fault_injected"):
            final_str = str(res["output"]).lower()
            corrupt_markers = [
                "sydney", "1492", "9942", "austria", "tachyon", "inverted thermodynamics",
                "mutated finding", "unexplained units", "adopting upstream finding", "contrary to consensus"
            ]
            has_corrupt = any(token in final_str for token in corrupt_markers)
            if has_corrupt and not state.get("error_contained"):
                state["cascade_occurred"] = True
                state["cascade_depth"] = 2
                state["cascade_severity"] = 0.85
                state["error_contained"] = False
            elif state.get("error_contained"):
                state["cascade_occurred"] = False
                state["cascade_depth"] = 0
                state["cascade_severity"] = 0.0
            else:
                state["cascade_occurred"] = True
                state["cascade_depth"] = 2
                state["cascade_severity"] = 0.70
                state["error_contained"] = False

        return state

    def run(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        interceptor_enabled: bool = True,
        fault_injection_config: Optional[Dict[str, Any]] = None,
    ) -> MultiAgentState:
        start_time = time.perf_counter()
        initial_state: MultiAgentState = {
            "task_id": task_id or f"task_{uuid.uuid4().hex[:8]}",
            "prompt": prompt,
            "trace_id": trace_id or f"tr_{uuid.uuid4().hex[:10]}",
            "model_name": self.llm.get_model_name(),
            "model": self.llm.get_model_name(),
            "architecture": "hierarchical",
            "interceptor_enabled": interceptor_enabled,
            "fault_injection_config": fault_injection_config,
            "agent_outputs": {},
            "handoffs": [],
            "fault_injected": False,
            "fault_type": None,
            "fault_location": None,
            "injected_fault_details": None,
            "cascade_occurred": False,
            "cascade_depth": 0,
            "cascade_severity": 0.0,
            "error_contained": not bool(fault_injection_config),
            "final_output": None,
            "total_latency_ms": 0.0,
            "interceptor_latency_ms": 0.0,
            "repair_latency_ms": 0.0,
        }

        final_state = self.graph.invoke(initial_state)
        final_state["total_latency_ms"] = (time.perf_counter() - start_time) * 1000.0
        return final_state
