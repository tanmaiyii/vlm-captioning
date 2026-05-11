"""
Pipeline orchestration: wire Stages 1-4 into a streaming Ray Data DAG.

How Ray Data streaming works (the concept behind this file)
-----------------------------------------------------------
When you call map/flat_map/map_batches on a Dataset, nothing executes.
You're building a *logical plan* — a DAG of transformations. Execution
only starts when a terminal operation fires (write_parquet here).

At execution time, Ray Data breaks the dataset into *blocks* (chunks of
rows) and pipelines them through all stages concurrently:

    Block 1: [fetch] → [decode] → [caption] → [write]
    Block 2:           [fetch]  → [decode]  → [caption] → [write]
    Block 3:                      [fetch]   → [decode]  → ...

While Block 1 is being captioned on the GPU, Block 2 is being decoded
on CPU, Block 3 is being fetched from S3. This is what "streaming by
default" means — stages overlap in time rather than running sequentially.

Consequence for GPU utilization
--------------------------------
Without streaming: all CPUs decode the full dataset → GPUs idle → GPUs
caption → CPUs idle. Spiky utilization, long wall clock.

With streaming: by the time the first blocks are decoded, the GPUs
start captioning while the CPUs keep decoding. Steady-state: both
T4s busy for most of the run (~80-90% utilization in the dashboard).

Why no .materialize() between stages
--------------------------------------
.materialize() forces Ray Data to fully execute all prior stages and
store results in the object store before continuing. Calling it between
stages reverts to bulk-sync: Stage N completes entirely before N+1
starts, leaving GPUs idle while CPUs decode and vice versa.

The notebook uses .materialize() between stages — that's a development
convenience for inspecting intermediate datasets (schema, row count,
sample row). This file does NOT, because production pipelines should
stay streaming. Same modules, different execution patterns.

Heterogeneous CPU/GPU scheduling (per the spec's requirement)
-------------------------------------------------------------
CPU stages request num_cpus=1, num_gpus=0 (default). GPU stage requests
num_gpus=1. Only GPU worker nodes have GPUs, so caption actors place
there exclusively. Decode lands on CPU workers because their 8 vCPU
have far more free capacity than the GPU workers' 4 vCPU (which are
mostly consumed by caption preprocessing).

For hard placement guarantees, --use-node-tags adds explicit
resources={"cpu_node": 1} / resources={"gpu_node": 1} per the spec's
own ray start hint.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ray

from velvet.captioner import caption_with_transformers, caption_with_vllm
from velvet.config import PipelineConfig
from velvet.decode import decode_and_sample
from velvet.io import read_manifest, write_captions

log = logging.getLogger(__name__)


def run(
    manifest_path: str,
    output_path: str,
    cfg: PipelineConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Execute the four-stage pipeline. Returns benchmark stats.

    Why no `.materialize()` between stages
    ---------------------------------------
    Ray Data's default execution is streaming — Stage N+1 starts as
    soon as Stage N emits its first batch. `.materialize()` forces
    sequential bulk execution: GPUs sit idle while CPUs decode the
    entire dataset, then CPUs sit idle while GPUs caption. The whole
    point of Ray Data over a script-with-queues is the streaming
    executor; opting out negates the framework choice.

    The notebook uses `.materialize()` in places — that's a development
    convenience for inspecting intermediate datasets, not a production
    pattern. This function does NOT.
    """
    t_start = time.time()

    # Stage 1: source (CSV manifest → fetched bytes)
    ds = read_manifest(manifest_path, limit=limit)

    # Stage 2: decode + sample (CPU stateless function).
    # Explicit num_cpus=1 matches the spec's hint and makes the
    # resource accounting visible in the Ray Dashboard.
    decode_kwargs: dict[str, Any] = {"num_cpus": 1}
    if cfg.use_node_tags:
        decode_kwargs["resources"] = {"cpu_node": 1}
    ds = ds.flat_map(decode_and_sample, **decode_kwargs)

    # Stage 3: caption (GPU stateful actor or vLLM engine)
    if cfg.backend == "transformers":
        log.info("backend=transformers")
        ds = caption_with_transformers(ds, cfg)
    elif cfg.backend == "vllm":
        log.info("backend=vllm")
        ds = caption_with_vllm(ds, cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend!r}")

    # Stage 4: sink
    write_captions(ds, output_path)

    elapsed = time.time() - t_start
    stats = {
        "backend": cfg.backend,
        "elapsed_sec": elapsed,
        "output": output_path,
        "batch_size": cfg.batch_size,
        "gpu_concurrency": cfg.gpu_concurrency,
        "limit": limit,
    }
    log.info(
        "pipeline_complete backend=%s elapsed_sec=%.1f output=%s",
        cfg.backend, elapsed, output_path,
    )
    return stats
