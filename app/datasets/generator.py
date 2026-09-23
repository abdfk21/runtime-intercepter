"""Automated benchmark task and trace dataset generator."""

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.datasets.schema import BenchmarkTask, TaskCategory, TraceRecord, HumanAnnotation, AnnotationLabel
from app.fault_injection.injector import FaultInjector
from app.fault_injection.fault_types import FaultType
from app.core.config import settings
from app.core.logging import logger


class DatasetGenerator:
    """Generates rigorous academic benchmark tasks and calibration/validation traces."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.fault_injector = FaultInjector(seed=seed)

    def generate_tasks(self, count: int = 80) -> List[BenchmarkTask]:
        """Generate diverse benchmark tasks across all 8 categories."""
        templates = {
            TaskCategory.FACTUAL_QA: [
                ("What is the capital city of Australia and when was it founded?", "Canberra was founded in 1913 as the federal capital.", ["Constitution 1901 Section 125", "Canberra foundation 1913"]),
                ("What is the largest planet in our solar system and its mass relative to Earth?", "Jupiter is the largest planet, over 300 times Earth's mass.", ["NASA Planetary Data", "Diameter 142,984 km"]),
                ("Who discovered penicillin and in what year was it announced?", "Alexander Fleming discovered penicillin in September 1928.", ["St. Mary's Hospital London", "1928 paper"]),
                ("What is the freezing point of heavy water (deuterium oxide) at 1 atm?", "Heavy water freezes at approximately 3.82 degrees Celsius.", ["NIST physical constants"]),
            ],
            TaskCategory.SUMMARIZATION: [
                ("Summarize the primary mechanism of mRNA vaccines and immunological memory.", "mRNA instructs cells to produce spike proteins triggering antigen-specific antibody production.", ["Ribosomal translation", "Memory B-cell induction"]),
                ("Summarize the core trade-offs between ACID and BASE database consistency models.", "ACID guarantees strict consistency and isolation; BASE prioritizes availability and eventual consistency.", ["CAP theorem", "Brewer 2000"]),
            ],
            TaskCategory.PLANNING: [
                ("Formulate an orbital insertion sequence for an interplanetary Mars probe.", "De-orbit burn, atmospheric aerobraking, retro-propulsive final capture into elliptical orbit.", ["Orbital mechanics delta-v budget"]),
                ("Outline a reproducible multi-stage validation pipeline for automated clinical models.", "Data sanitization, prospective cohort partitioning, leakage testing, external site validation.", ["Good Machine Learning Practice Guidelines"]),
            ],
            TaskCategory.MULTI_STEP_REASONING: [
                ("If a car travels 120 km at 60 km/h, then 120 km at 40 km/h, what is the average speed?", "Total distance 240 km, total time 5 hours (2h + 3h), average speed is 48 km/h.", ["Harmonic mean calculation"]),
                ("Determine which element has atomic number 26 and explain its electron configuration.", "Iron (Fe) has atomic number 26; configuration is [Ar] 3d6 4s2.", ["Periodic table of elements"]),
            ],
            TaskCategory.INFORMATION_EXTRACTION: [
                ("Extract all quantitative clinical endpoints from the trial abstract protocol.", "Primary endpoint: progression-free survival (hazard ratio 0.68). Secondary endpoint: overall response rate (42%).", ["Clinical trial registry protocol NCT04192"]),
                ("Extract all governing regulatory statutes referenced in the financial compliance filing.", "Sarbanes-Oxley Section 404, Dodd-Frank Section 165, and Basel III capital adequacy framework.", ["SEC 10-K disclosures"]),
            ],
            TaskCategory.COMPARISON: [
                ("Compare Transformer attention complexity with state-space models like Mamba.", "Standard self-attention scales quadratically O(N^2); state-space models scale linearly O(N) during sequence processing.", ["Vaswani et al. 2017", "Gu & Dao 2023"]),
                ("Compare synchronous Byzantine Fault Tolerance with asynchronous Paxos under network partition.", "BFT tolerates up to 1/3 Byzantine adversarial nodes; Paxos tolerates up to 1/2 crash failures under partition.", ["Lamport Paxos 1998", "Castro & Liskov 1999"]),
            ],
            TaskCategory.NUMERICAL_REASONING: [
                ("Calculate compound interest on $10,000 at 5% annual rate compounded quarterly for 3 years.", "Formula: A = 10000*(1 + 0.05/4)^(4*3) = $11,607.55.", ["Compound interest formula"]),
                ("Compute the Shannon entropy of a fair 8-sided die in bits.", "Entropy H = log2(8) = 3 bits per roll.", ["Information theory fundamentals"]),
            ],
            TaskCategory.RESEARCH_SYNTHESIS: [
                ("Synthesize current literature on hallucination containment in autonomous multi-agent pipelines.", "Multi-agent error cascading requires runtime boundary monitoring, self-correction loops, and cross-agent consistency verification.", ["Survey on Multi-Agent LLMs 2024", "Cascade Mitigation Studies"]),
                ("Synthesize architectural differences between centralized hierarchical and decentralized consensus agents.", "Hierarchical structures bottleneck at the coordinator; decentralized consensus scales validation but incurs latency.", ["Multi-Agent Architecture Taxonomy 2024"]),
            ],
        }

        tasks: List[BenchmarkTask] = []
        task_idx = 1

        for category, cat_templates in templates.items():
            needed = count // len(templates)
            for i in range(needed):
                tmpl = cat_templates[i % len(cat_templates)]
                task = BenchmarkTask(
                    task_id=f"bench_{category.value}_{task_idx:04d}",
                    category=category,
                    prompt=tmpl[0] + (f" (Variation #{i+1})" if i >= len(cat_templates) else ""),
                    difficulty="easy" if i % 3 == 0 else ("medium" if i % 3 == 1 else "hard"),
                    ground_truth=tmpl[1],
                    reference_facts=tmpl[2],
                    metadata={"template_idx": i, "category": category.value},
                )
                tasks.append(task)
                task_idx += 1

        return tasks

    def generate_validation_dataset(
        self,
        tasks: List[BenchmarkTask],
        target_size: int = 350,
        corruption_rate: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Generates 300–400 trace validation dataset containing clean and fault-injected traces."""
        traces: List[Dict[str, Any]] = []
        fault_types_list = list(FaultType)
        models = ["deterministic-mock-v1", "gemini-2.5-flash", "llama-3.3-70b-versatile"]
        architectures = ["sequential", "hierarchical", "parallel_consensus"]

        for i in range(target_size):
            task = tasks[i % len(tasks)]
            model = models[i % len(models)]
            arch = architectures[i % len(architectures)]
            trace_id = f"trace_val_{i+1:04d}"
            
            # Ground truth determination: clean vs corrupted
            is_corrupted = (self.rng.random() < corruption_rate)
            fault_type = self.rng.choice(fault_types_list) if is_corrupted else None

            # Base clean structured output
            base_output = {
                "answer": f"Verified analysis addressing: {task.prompt[:60]}.",
                "claims": [
                    {
                        "claim": task.reference_facts[0] if task.reference_facts else "Primary ground truth claim.",
                        "confidence": round(self.rng.uniform(0.88, 0.98), 2),
                        "evidence": [f"Verified reference: {task.ground_truth[:40]}"]
                    },
                    {
                        "claim": task.reference_facts[1] if len(task.reference_facts) > 1 else "Corroborated analytical property.",
                        "confidence": round(self.rng.uniform(0.82, 0.94), 2),
                        "evidence": ["Systematic validation registry."]
                    }
                ],
                "assumptions": ["Standard domain constraints apply."],
                "sources": ["Empirical Dataset Repository v1.0"]
            }

            injected_details = None
            if is_corrupted and fault_type:
                fi_res = self.fault_injector.inject_fault(
                    output=base_output,
                    fault_type=fault_type,
                    location="agent_handoff",
                    severity=0.85,
                    seed=self.seed + i,
                )
                source_output = fi_res.corrupted_output
                injected_details = fi_res.to_dict()
                label = AnnotationLabel.HALLUCINATED if fault_type in [FaultType.FACTUAL_MUTATION, FaultType.NUMERICAL_MUTATION, FaultType.UNSUPPORTED_CLAIM_INSERTION] else (
                    AnnotationLabel.CONTRADICTORY if fault_type == FaultType.CONTRADICTION_INSERTION else AnnotationLabel.SUSPICIOUS
                )
            else:
                source_output = base_output
                label = AnnotationLabel.VALID

            record = {
                "trace_id": trace_id,
                "task_id": task.task_id,
                "task_category": task.category.value,
                "prompt": task.prompt,
                "architecture": arch,
                "model": model,
                "is_corrupted": is_corrupted,
                "fault_type": fault_type.value if fault_type else None,
                "source_output": source_output,
                "target_context": {"prompt": task.prompt, "ground_truth": task.ground_truth},
                "injected_fault_details": injected_details,
                "ground_truth_label": label.value,
                "is_synthetic_development_label": True,  # Explicitly marked per Section 19 rule
                "annotator_id": "SYNTHETIC_DEVELOPMENT_GENERATOR",
                "notes": "Automated synthetic development trace for calibration and unit verification.",
            }
            traces.append(record)

        return traces

    def save_dataset(self, data: List[Dict[str, Any]], filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved dataset of {len(data)} items to {filepath}")
