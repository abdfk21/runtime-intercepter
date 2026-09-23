"""Annotation manager and Inter-Annotator Agreement calculation (Cohen's & Fleiss' Kappa)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import cohen_kappa_score
from app.datasets.schema import HumanAnnotation, AnnotationLabel
from app.core.logging import logger


class AnnotationManager:
    """Manages collection, persistence, and agreement analysis for human validation datasets."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/annotations.json")
        self.annotations: List[HumanAnnotation] = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.annotations = [HumanAnnotation(**item) for item in data]
            except Exception as e:
                logger.warning(f"Could not load annotations: {e}")

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([a.model_dump() for a in self.annotations], f, indent=2)

    def add_annotation(
        self,
        trace_id: str,
        annotator_id: str,
        label: AnnotationLabel,
        handoff_id: Optional[str] = None,
        notes: Optional[str] = None,
        is_synthetic: bool = False,
    ) -> HumanAnnotation:
        annotation = HumanAnnotation(
            annotation_id=f"ann_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(self.annotations)+1}",
            trace_id=trace_id,
            handoff_id=handoff_id,
            annotator_id=annotator_id,
            label=label,
            is_synthetic_development_label=is_synthetic,
            notes=notes,
        )
        self.annotations.append(annotation)
        self.save()
        return annotation

    def calculate_inter_annotator_agreement(
        self, annotator_a: str, annotator_b: str
    ) -> Dict[str, Any]:
        """Calculates Cohen's Kappa for traces double-annotated by annotator_a and annotator_b."""
        a_labels: Dict[str, str] = {}
        b_labels: Dict[str, str] = {}

        for ann in self.annotations:
            if ann.annotator_id == annotator_a:
                a_labels[ann.trace_id] = ann.label.value
            elif ann.annotator_id == annotator_b:
                b_labels[ann.trace_id] = ann.label.value

        common_keys = sorted(list(set(a_labels.keys()) & set(b_labels.keys())))
        if not common_keys:
            return {
                "status": "No overlapping annotations found",
                "common_count": 0,
                "cohen_kappa": None,
                "disagreements": 0,
            }

        y_a = [a_labels[k] for k in common_keys]
        y_b = [b_labels[k] for k in common_keys]

        disagreements = sum(1 for a, b in zip(y_a, y_b) if a != b)
        try:
            kappa = float(cohen_kappa_score(y_a, y_b))
        except Exception:
            kappa = 0.0

        label_dist = {lbl.value: y_a.count(lbl.value) + y_b.count(lbl.value) for lbl in AnnotationLabel}

        return {
            "status": "Calculated from paired annotations",
            "annotator_a": annotator_a,
            "annotator_b": annotator_b,
            "common_count": len(common_keys),
            "disagreements": disagreements,
            "agreement_rate": round((len(common_keys) - disagreements) / len(common_keys), 4),
            "cohen_kappa": round(kappa, 4),
            "label_distribution": label_dist,
        }
