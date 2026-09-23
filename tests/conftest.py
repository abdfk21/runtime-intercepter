"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.mock import MockProvider
from app.interceptor.interceptor import RuntimeInterceptor
from app.fault_injection.injector import FaultInjector
from app.storage.database import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialize database for test runs."""
    init_db()


@pytest.fixture
def mock_llm():
    return MockProvider(seed=42)


@pytest.fixture
def interceptor():
    return RuntimeInterceptor()


@pytest.fixture
def fault_injector():
    return FaultInjector(seed=42)


@pytest.fixture
def sample_valid_output():
    return {
        "answer": "Canberra is the federal capital of Australia, established in 1913.",
        "claims": [
            {
                "claim": "Canberra is the capital city of Australia.",
                "confidence": 0.98,
                "evidence": ["Australian Constitution Section 125."],
            },
            {
                "claim": "Canberra was founded in 1913.",
                "confidence": 0.95,
                "evidence": ["Official foundation records 1913."],
            },
        ],
        "assumptions": ["Standard geographic sovereign definitions."],
        "sources": ["National Archives of Australia"],
    }


@pytest.fixture
def sample_corrupted_output():
    return {
        "answer": "The capital of Australia is Sydney.",
        "claims": [
            {
                "claim": "The capital of Australia is Sydney.",
                "confidence": 0.99,
                "evidence": ["Census data"],
            },
            {
                "claim": "All quantum teleportation experiments succeeded with 994200 units.",
                "confidence": 1.0,
                "evidence": [],
            },
        ],
        "assumptions": ["Inverted thermodynamics laws apply."],
        "sources": ["Anonymous internet blog"],
    }
