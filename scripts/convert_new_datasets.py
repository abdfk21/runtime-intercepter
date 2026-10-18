"""Convert newly provided external datasets into the project’s benchmark trace schema.

This script reads the archives under `new made datasets/` and normalizes them into a
list of records compatible with `app.datasets.schema.TraceRecord`-style data used by the
validation/calibration pipeline.

It supports:
- raw JSONL corpus files such as HaluEval / RAGTruth / FEVER-style outputs
- nested synthetic trace archives produced by the project bootstrap pipeline
- generic JSON/CSV record sets with prompt + answer / hallucinated output fields
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "new made datasets"
OUTPUT_DIR = ROOT / "data" / "processed" / "converted_external_datasets"


def safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def infer_category(filename: str, record: Dict[str, Any]) -> str:
    name = filename.lower()
    if "dialogue" in name or "qa" in name:
        return "factual_qa"
    if "summar" in name:
        return "summarization"
    if "research" in name:
        return "research_synthesis"
    if "comparison" in name:
        return "comparison"
    if "reason" in name:
        return "multi_step_reasoning"
    if "fever" in name or "fact" in name:
        return "factual_qa"
    if "ragtruth" in name:
        return "research_synthesis"
    return record.get("task_category") or "factual_qa"


def parse_json_lines(raw_text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def parse_json_blob(raw_bytes: bytes) -> List[Dict[str, Any]]:
    text = raw_bytes.decode("utf-8", errors="replace")
    if not text.strip():
        return []

    if text.lstrip().startswith("["):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [p for p in payload if isinstance(p, dict)]
        except json.JSONDecodeError:
            pass

    if text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return [payload]
        except json.JSONDecodeError:
            pass

    return parse_json_lines(text)


def parse_csv_blob(raw_bytes: bytes) -> List[Dict[str, Any]]:
    text = raw_bytes.decode("utf-8", errors="replace")
    if not text.strip():
        return []

    rows: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rows.append({k: v for k, v in row.items() if v is not None})
    return rows


def iter_archive_members(path: Path) -> Iterator[Tuple[str, bytes]]:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            yield name, zf.read(name)


def deep_iter_archive_members(path: Path) -> Iterator[Tuple[str, bytes]]:
    seen: set[str] = set()

    def walk_zip(current_path: Path, archive_name: str = "") -> Iterator[Tuple[str, bytes]]:
        for member_name, member_bytes in iter_archive_members(current_path):
            full_name = f"{archive_name}/{member_name}" if archive_name else member_name
            if full_name in seen:
                continue
            seen.add(full_name)
            if member_name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(member_bytes)) as inner:
                    for inner_name in inner.namelist():
                        if inner_name.endswith("/"):
                            continue
                        yield f"{full_name}/{inner_name}", inner.read(inner_name)
                continue
            yield full_name, member_bytes

    yield from walk_zip(path)


def pick_first(mapping: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        text = safe_text(value)
        if text:
            return text
    return None


def convert_synthetic_trace_record(record: Dict[str, Any], source_name: str, index: int) -> Dict[str, Any]:
    steps = record.get("steps") or []
    seed_question = safe_text(record.get("seed_question")) or safe_text(record.get("prompt")) or ""
    last_step = steps[-1] if isinstance(steps, list) and steps else {}
    last_output = safe_text(last_step.get("output")) or ""

    agent_outputs: Dict[str, Any] = {}
    handoffs: List[Dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        agent_name = str(step.get("agent") or "agent")
        step_output = safe_text(step.get("output")) or ""
        target_agent = step.get("handoff_to")
        agent_outputs[agent_name] = {
            "answer": step_output,
            "claims": [{"claim": step_output, "confidence": 0.9, "evidence": ["Converted from synthetic trace"]}],
            "assumptions": ["Converted from synthetic trace."],
            "sources": [source_name],
        }
        if target_agent:
            handoffs.append({
                "source_agent": agent_name,
                "target_agent": str(target_agent),
                "source_output": step_output,
                "decision": "PASS",
                "risk_score": 0.0,
                "threshold": 0.5,
                "explanations": ["Converted synthetic trace handoff."],
            })

    final_output = {
        "answer": last_output,
        "claims": [{"claim": last_output, "confidence": 0.9, "evidence": ["Converted from synthetic trace"]}],
        "assumptions": ["Converted from synthetic trace."],
        "sources": [source_name],
    }

    corrupted = bool(record.get("corrupted"))
    fault_type = record.get("fault_injected") or record.get("fault_type")
    if not fault_type and corrupted:
        fault_type = "factual_mutation"

    return {
        "trace_id": str(record.get("trace_id") or f"synthetic_{index:04d}"),
        "task_id": str(record.get("task_id") or f"task_{source_name}_{index:04d}"),
        "task_category": record.get("task_category") or "factual_qa",
        "architecture": str(record.get("architecture") or "unknown"),
        "model": str(record.get("model") or "unknown"),
        "condition": "INTERCEPTOR_ON" if corrupted else "INTERCEPTOR_OFF",
        "prompt": seed_question,
        "source_output": final_output,
        "agent_outputs": agent_outputs or {"researcher": final_output},
        "handoffs": handoffs,
        "final_output": final_output,
        "fault_injected": corrupted,
        "fault_type": fault_type,
        "injected_fault_details": {
            "corrupted_step_index": record.get("corrupted_step_index"),
            "fault_type": fault_type,
            "source": source_name,
        } if corrupted else None,
        "cascade_occurred": False,
        "cascade_depth": 0,
        "cascade_severity": 0.0,
        "error_contained": False,
        "total_latency_ms": 0.0,
        "interceptor_latency_ms": 0.0,
        "repair_latency_ms": 0.0,
        "ground_truth_label": "HALLUCINATED" if corrupted else "VALID",
        "is_synthetic_development_label": True,
        "annotator_id": "EXTERNAL_DATASET_CONVERTER",
        "notes": f"Converted from synthetic trace archive: {source_name}",
        "metadata": {
            "source_file": source_name,
            "raw_record": {k: v for k, v in record.items() if k not in {"steps"}},
        },
    }


def normalize_record(record: Dict[str, Any], source_name: str, index: int) -> Dict[str, Any]:
    if isinstance(record.get("steps"), list):
        return convert_synthetic_trace_record(record, source_name, index)

    record = dict(record)
    prompt = pick_first(record, "prompt", "question", "query", "input", "dialogue_history", "instruction", "task", "seed_question")
    ground_truth = pick_first(
        record,
        "ground_truth",
        "answer",
        "right_response",
        "correct_response",
        "target",
        "gold",
        "label",
        "expected_answer",
    )
    source_output = pick_first(
        record,
        "source_output",
        "generated_output",
        "model_output",
        "response",
        "hallucinated_response",
        "predicted_answer",
        "output",
    )

    if source_output is None and "hallucinated_response" in record:
        source_output = safe_text(record.get("hallucinated_response"))
    if ground_truth is None and "right_response" in record:
        ground_truth = safe_text(record.get("right_response"))

    if prompt is None:
        prompt = pick_first(record, "context", "knowledge", "text")

    task_category = record.get("task_category") or infer_category(source_name, record)
    architecture = record.get("architecture") or "unknown"
    model = record.get("model") or "unknown"
    condition = record.get("condition") or "INTERCEPTOR_OFF"

    is_corrupted = bool(record.get("is_corrupted")) or bool(record.get("fault_injected"))
    if not is_corrupted and ground_truth and source_output and source_output != ground_truth:
        is_corrupted = True

    fault_type = record.get("fault_type")
    if not fault_type:
        if "numerical" in str(record).lower():
            fault_type = "numerical_mutation"
        elif "contradict" in str(record).lower():
            fault_type = "contradiction_insertion"
        elif "unsupported" in str(record).lower() or "fabricated" in str(record).lower():
            fault_type = "unsupported_claim_insertion"
        elif is_corrupted:
            fault_type = "factual_mutation"

    final_output = {
        "answer": source_output or ground_truth or "",
        "claims": [
            {
                "claim": ground_truth or source_output or "",
                "confidence": 0.9,
                "evidence": ["Converted from external dataset"],
            }
        ],
        "assumptions": ["Converted from external dataset."],
        "sources": [source_name],
    }

    normalized = {
        "trace_id": str(record.get("trace_id") or f"external_{source_name}_{index:04d}"),
        "task_id": str(record.get("task_id") or f"task_{source_name}_{index:04d}"),
        "task_category": task_category,
        "architecture": architecture,
        "model": model,
        "condition": condition,
        "prompt": prompt or "",
        "source_output": final_output,
        "agent_outputs": {
            "researcher": final_output,
            "specialist": final_output,
            "verifier": final_output,
            "critic": final_output,
            "synthesizer": final_output,
        },
        "handoffs": [],
        "final_output": final_output,
        "fault_injected": is_corrupted,
        "fault_type": fault_type,
        "injected_fault_details": record.get("injected_fault_details") or (None if not is_corrupted else {"source": source_name, "fault_type": fault_type}),
        "cascade_occurred": False,
        "cascade_depth": 0,
        "cascade_severity": 0.0,
        "error_contained": False,
        "total_latency_ms": 0.0,
        "interceptor_latency_ms": 0.0,
        "repair_latency_ms": 0.0,
        "ground_truth_label": "HALLUCINATED" if is_corrupted else "VALID",
        "is_synthetic_development_label": True,
        "annotator_id": "EXTERNAL_DATASET_CONVERTER",
        "notes": f"Converted from {source_name}",
        "metadata": {
            "source_file": source_name,
            "raw_record": {k: v for k, v in record.items() if k not in {"prompt", "question"}},
        },
    }
    return normalized


def parse_member_records(member_name: str, member_bytes: bytes) -> List[Dict[str, Any]]:
    name_lower = member_name.lower()

    if name_lower.endswith(".jsonl"):
        return parse_json_lines(member_bytes.decode("utf-8", errors="replace"))
    if name_lower.endswith(".json"):
        return parse_json_blob(member_bytes)
    if name_lower.endswith(".csv"):
        return parse_csv_blob(member_bytes)
    return []


def process_dataset_archive(archive_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for member_name, member_bytes in deep_iter_archive_members(archive_path):
        name_lower = member_name.lower()
        if not any(name_lower.endswith(ext) for ext in [".json", ".jsonl", ".csv"]):
            continue
        if "readme" in name_lower or "clean_datasets" in name_lower:
            continue
        try:
            parsed = parse_member_records(member_name, member_bytes)
        except Exception:
            continue
        for idx, rec in enumerate(parsed):
            if not isinstance(rec, dict):
                continue
            records.append(normalize_record(rec, archive_path.name, len(records) + idx + 1))
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATASET_ROOT}")

    archive_paths = sorted(DATASET_ROOT.glob("*.zip"))
    if not archive_paths:
        raise FileNotFoundError(f"No zip datasets were found in {DATASET_ROOT}")

    summary: List[Dict[str, Any]] = []
    for archive_path in archive_paths:
        converted = process_dataset_archive(archive_path)
        out_path = OUTPUT_DIR / f"{archive_path.stem}_converted.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        summary.append({
            "source": archive_path.name,
            "output": str(out_path.relative_to(ROOT)),
            "count": len(converted),
        })

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
