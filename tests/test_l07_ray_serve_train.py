"""L07 Ray Serve / Ray Train smokes on CPU with a fake or tiny model (slow: ~30-60 s each)."""

from __future__ import annotations

import asyncio

import pytest

ray = pytest.importorskip("ray")


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_serve_fake_deployment_and_closed_loop_client():
    from ray import serve

    from distributed.ray_serve.bench import closed_loop
    from distributed.ray_serve.deployment import build_app, start

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    try:
        app = build_app(None, replicas=1, max_ongoing=4, max_new_tokens=4, fake_ms=5, proxy=None, batch_wait_ms=5, max_batch=4, gpu_frac=0)
        url = start(app, port=8011)
        recs = asyncio.run(closed_loop(url, [f"p{i}" for i in range(8)], concurrency=4, max_tokens=4))
        assert len(recs) == 8 and all(r.success for r in recs)
        assert any((r.extra.get("batch") or 1) > 1 for r in recs)  # Serve-level batching kicked in
    finally:
        serve.shutdown()
        ray.shutdown()


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_ray_train_two_cpu_workers_end_with_identical_params(tmp_path):
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from distributed.ray_train.torch_trainer import train_loop_per_worker

    cfg = {"vocab": 64, "seq_len": 16, "d_model": 32, "n_layers": 1, "n_samples": 64, "batch_size": 8, "epochs": 1, "lr": 1e-3, "seed": 0}
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    try:
        trainer = TorchTrainer(train_loop_per_worker, train_loop_config=cfg, scaling_config=ScalingConfig(num_workers=2, use_gpu=False), run_config=RunConfig(name="t", storage_path=str(tmp_path)))
        result = trainer.fit()
    finally:
        ray.shutdown()
    assert result.metrics["loss"] > 0 and result.metrics["world"] == 2
    assert result.metrics["params_identical_across_workers"] is True
    assert result.checkpoint is not None
