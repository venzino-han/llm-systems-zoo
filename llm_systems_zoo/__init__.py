"""llm-systems-zoo: shared benchmark harness.

Everything in this package must import without torch, CUDA, or network access so that
the simulators, calculators and analysis code can run in CPU-only CI.
"""

__version__ = "0.1.0"

from .gpu import free_memory_gib, wait_for_free_memory  # noqa: F401
from .manifest import Run, RunManifest, config_hash  # noqa: F401
from .metrics import SLO, RequestRecord, goodput, latency_summary, summarize, throughput  # noqa: F401
from .results import copy_results  # noqa: F401
from .timer import PhaseTimer, cuda_sync  # noqa: F401
