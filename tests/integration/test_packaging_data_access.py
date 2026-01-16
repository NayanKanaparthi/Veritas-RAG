"""Test that benchmark datasets are accessible after package installation."""

import json

import pytest
from importlib import resources


def test_eval_set_accessible():
    """Assert that eval_set_v1.jsonl can be located and read via importlib.resources."""
    try:
        data_path = resources.files("veritas_rag.benchmarks.data").joinpath("eval_set_v1.jsonl")
        with resources.as_file(data_path) as p:
            assert p.exists()
            # Read first line to verify it's valid JSONL
            with open(p, "r") as f:
                first_line = f.readline()
                assert first_line.strip()
                json.loads(first_line)  # Should parse
    except (ModuleNotFoundError, FileNotFoundError) as e:
        pytest.fail(f"Eval set not accessible: {e}")
