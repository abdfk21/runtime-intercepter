"""Experiment Manager for automated multi-agent benchmarking and evaluation."""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.llm.factory import LLMFactory
from app.interceptor.interceptor import RuntimeInterceptor
from app.fault_injection.injector import FaultInjector
from app.fault_injection.fault_types import FaultType
from app.orchestration.sequential import SequentialPipeline
from app.orchestration.hierarchical import HierarchicalPipeline
from app.orchestration.parallel_consensus import ParallelConsensusPipeline
from app.datasets.generator import DatasetGenerator
from app.datasets.loader import DatasetLoader
from app.evaluation.metrics import EvaluationMetrics
from app.experiments.reproducibility import ReproducibilityTracker
from app.storage.database import SessionLocal
from app.storage.repository import Repository


class ExperimentManager:
    """Coordinates benchmark execution across models, architectures, and experimental conditions."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        ReproducibilityTracker.set_deterministic_seeds(seed)
        self.dataset_gen = DatasetGenerator(seed=seed)
        self.fault_injector = FaultInjector(seed=seed)

    def run_experiment(
        self,
        experiment_id: Optional[str] = None,
        models: Optional[List[str]] = None,
        architectures: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        fault_types: Optional[List[str]] = None,
        num_tasks: int = 12,
        use_adaptive_calibration: bool = True,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        models = models or ["mock"]
        architectures = architectures or ["sequential", "hierarchical", "parallel_consensus"]
        conditions = conditions or ["INTERCEPTOR_ON", "INTERCEPTOR_OFF"]
        fault_types = fault_types or [
            "factual_mutation",
            "numerical_mutation",
            "contradiction_insertion",
            "unsupported_claim_insertion",
        ]

        logger.info(f"Initiating Experiment {exp_id} with {len(models)} models, {len(architectures)} architectures, {num_tasks} tasks.")

        converted_tasks = DatasetLoader.load_converted_external_tasks()
        if converted_tasks:
            tasks = converted_tasks[:max(1, min(num_tasks, len(converted_tasks)))]
            logger.info(f"Using {len(tasks)} converted external dataset tasks for experiment {exp_id}.")
        else:
            tasks = self.dataset_gen.generate_tasks(count=max(20, num_tasks))[:num_tasks]

        all_traces: List[Dict[str, Any]] = []
        metadata = ReproducibilityTracker.capture_environment_metadata(self.seed)

        db = SessionLocal() if save_to_db else None
        repo = Repository(db) if db else None

        try:
            for m_name in models:
                # LLM creation with fallback transparency
                llm, prov_meta = LLMFactory.create(
                    provider_name=m_name, allow_fallback=True, seed=self.seed
                )

                interceptor = RuntimeInterceptor(
                    use_adaptive_calibration=use_adaptive_calibration
                )

                pipelines = {
                    "sequential": SequentialPipeline(llm_provider=llm, interceptor=interceptor, fault_injector=self.fault_injector),
                    "hierarchical": HierarchicalPipeline(llm_provider=llm, interceptor=interceptor, fault_injector=self.fault_injector),
                    "parallel_consensus": ParallelConsensusPipeline(llm_provider=llm, interceptor=interceptor, fault_injector=self.fault_injector),
                }

                for arch_name in architectures:
                    pipeline = pipelines.get(arch_name)
                    if not pipeline:
                        continue

                    for cond in conditions:
                        interceptor_enabled = (cond == "INTERCEPTOR_ON")

                        if repo:
                            repo.save_experiment(
                                experiment_id=f"{exp_id}_{m_name}_{arch_name}_{cond}",
                                model_name=prov_meta["model_name"],
                                architecture=arch_name,
                                condition=cond,
                                random_seed=self.seed,
                                config={"requested_model": m_name, "provider_metadata": prov_meta},
                            )

                        for t_idx, task in enumerate(tasks):
                            # Deterministically assign fault type (half clean, half faulted)
                            inject_fault = (t_idx % 2 == 1)
                            ft = fault_types[t_idx % len(fault_types)] if inject_fault else None

                            fi_cfg = (
                                {
                                    "fault_type": ft,
                                    "location": "researcher" if arch_name != "parallel_consensus" else "researcher_a",
                                    "seed": self.seed + t_idx,
                                }
                                if inject_fault
                                else None
                            )

                            cond_tag = "ON" if "ON" in cond else "OFF"
                            trace_id = f"tr_{exp_id[:6]}_{m_name[:4]}_{arch_name[:3]}_{cond_tag}_{t_idx+1:03d}"
                            
                            trace_result = pipeline.run(
                                prompt=task.prompt,
                                task_id=task.task_id,
                                trace_id=trace_id,
                                interceptor_enabled=interceptor_enabled,
                                fault_injection_config=fi_cfg,
                            )
                            trace_result["experiment_id"] = f"{exp_id}_{m_name}_{arch_name}_{cond}"
                            trace_result["condition"] = cond
                            trace_result["task_category"] = task.category.value
                            all_traces.append(trace_result)

                            if repo:
                                repo.save_task(task.model_dump())
                                repo.save_trace(trace_result)

        finally:
            if db:
                db.close()

        # Compute summary metrics
        on_traces = [t for t in all_traces if t.get("condition") == "INTERCEPTOR_ON"]
        off_traces = [t for t in all_traces if t.get("condition") == "INTERCEPTOR_OFF"]

        on_metrics = EvaluationMetrics.calculate_cascade_metrics(on_traces)
        off_metrics = EvaluationMetrics.calculate_cascade_metrics(off_traces)

        summary = {
            "experiment_id": exp_id,
            "metadata": metadata,
            "total_traces_executed": len(all_traces),
            "interceptor_on_traces": len(on_traces),
            "interceptor_off_traces": len(off_traces),
            "on_metrics": on_metrics,
            "off_metrics": off_metrics,
            "traces": all_traces,
        }
        logger.info(
            f"Experiment {exp_id} complete. Containment Rate: ON={on_metrics.get('error_containment_rate', 0):.1%} "
            f"vs OFF={off_metrics.get('error_containment_rate', 0):.1%}"
        )
        return summary
