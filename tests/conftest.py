"""Test configuration.

Default `pytest` runs only CPU tests (what CI runs). GPU-only tests are selected with
`--gpu`; tests that download models or datasets are selected with `--slow`.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gpu", action="store_true", default=False, help="run tests that need a CUDA GPU")
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests (model/dataset downloads)")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_gpu = config.getoption("--gpu")
    run_slow = config.getoption("--slow")
    cuda = _cuda_available() if run_gpu else False
    skip_gpu = pytest.mark.skip(reason="needs --gpu and a CUDA device")
    skip_slow = pytest.mark.skip(reason="needs --slow")
    for item in items:
        if "gpu" in item.keywords and not (run_gpu and cuda):
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
