"""
Run benchmarks. Two modes:

  1. Compare backends:
       python benchmark.py --manifest manifest.csv --limit 200
     Runs both backends back-to-back with the same config. Emits
     `benchmark_results.json` and `benchmark_table.md`.

  2. Sweep configs (stretch goal — throughput tuning):
       python benchmark.py --manifest manifest.csv --limit 100 \
           --sweep --backend transformers
     Runs the same backend across multiple (batch_size, gpu_concurrency)
     pairs. Emits `sweep_results.json` and `sweep_table.md`. Useful for
     answering "what's the best batch_size for this T4?"

Implementation note
-------------------
We import `velvet.pipeline.run` directly rather than spawning a
subprocess of `run.py`. Direct calls give cleaner wall-clock timing
(no Python interpreter spinup overhead) and let the same Ray runtime
be reused across runs. The cost is that vLLM's process-level env
changes (like the now-removed `VLLM_ATTENTION_BACKEND` env var —
superseded by the `attention_backend` engine_kwarg in vLLM 0.20.2)
must be set before the
import — handled in `caption_with_vllm`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Make src/ importable when running the file directly.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC)

import ray  # noqa: E402

from velvet.config import PipelineConfig  # noqa: E402
from velvet.pipeline import run as run_pipeline  # noqa: E402

# Ship src/velvet/ to Ray workers so they can import velvet.*.
# Mirrors run.py — see the _init_ray docstring there for why.
ray.init(
    runtime_env={"py_modules": [os.path.join(_SRC, "velvet")]},
    ignore_reinit_error=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("benchmark")


def run_one(
    backend: str,
    manifest: str,
    output_root: str,
    limit: int | None,
    batch_size: int,
    gpu_concurrency: int,
) -> dict:
    """Run the pipeline once with the given backend, return stats."""
    cfg = PipelineConfig(
        backend=backend,
        batch_size=batch_size,
        gpu_concurrency=gpu_concurrency,
    )
    output = f"{output_root.rstrip('/')}/captions-{backend}"

    log.info("=" * 70)
    log.info("RUN backend=%s output=%s", backend, output)
    log.info("=" * 70)

    try:
        stats = run_pipeline(
            manifest_path=manifest,
            output_path=output,
            cfg=cfg,
            limit=limit,
        )
        stats["error"] = None
    except Exception as e:
        log.exception("backend=%s failed", backend)
        stats = {
            "backend": backend,
            "elapsed_sec": None,
            "output": output,
            "batch_size": batch_size,
            "limit": limit,
            "error": str(e),
        }
    return stats


def count_rows(parquet_path: str) -> int:
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        return len(df)
    except Exception:
        return -1


def render_table(results: list[dict]) -> str:
    """Produce the markdown table that goes into WRITEUP §4.3."""
    tf = next((r for r in results if r["backend"] == "transformers"), None)
    vl = next((r for r in results if r["backend"] == "vllm"), None)

    def cell(run, key, fmt="{}"):
        if run is None or run.get(key) is None:
            return "—"
        return fmt.format(run[key])

    lines = [
        "# Benchmark — transformers vs vLLM",
        "",
        f"Run on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "| Metric | transformers (HF) | vLLM (Ray Data LLM) | Notes |",
        "|---|---|---|---|",
        f"| Wall clock (s) | {cell(tf, 'elapsed_sec', '{:.1f}')} | {cell(vl, 'elapsed_sec', '{:.1f}')} | Lower is better |",
        f"| Output rows | {cell(tf, 'rows')} | {cell(vl, 'rows')} | Sanity — should match |",
        f"| Throughput (videos/s) | {cell(tf, 'throughput', '{:.2f}')} | {cell(vl, 'throughput', '{:.2f}')} | Higher is better |",
        f"| Batch size | {cell(tf, 'batch_size')} | {cell(vl, 'batch_size')} | Same in both runs |",
    ]

    if tf and vl and tf.get("elapsed_sec") and vl.get("elapsed_sec"):
        speedup = tf["elapsed_sec"] / vl["elapsed_sec"]
        winner = "vLLM faster" if speedup > 1 else "transformers faster"
        lines.append("")
        lines.append(f"**vLLM speedup: {speedup:.2f}x** ({winner})")

    return "\n".join(lines)


def render_sweep_table(results: list[dict]) -> str:
    """Render the throughput-tuning sweep as a markdown table."""
    lines = [
        "# Throughput tuning sweep",
        "",
        f"Run on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "| backend | batch_size | gpu_concurrency | wall (s) | throughput (vid/s) | rows | error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        wall = f"{r['elapsed_sec']:.1f}" if r.get("elapsed_sec") else "—"
        thru = f"{r['throughput']:.2f}" if r.get("throughput") else "—"
        err = (r.get("error") or "")[:40]
        lines.append(
            f"| {r['backend']} | {r['batch_size']} | {r['gpu_concurrency']} | "
            f"{wall} | {thru} | {r.get('rows', '—')} | {err} |"
        )
    lines.append("")
    lines.append("**Reading the table:** find the row with highest throughput "
                 "that completed without OOM. That's your sweet spot. The "
                 "WRITEUP discusses why we chose `batch_size=2, "
                 "gpu_concurrency=2` as the default starting point — the OOM "
                 "ceiling depends on the do_image_splitting setting in the "
                 "SmolVLM2 processor (see WRITEUP §Sizing and CHANGES.md "
                 "2026-05-11).")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-concurrency", type=int, default=2)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--skip", choices=["transformers", "vllm"], default=None,
        help="(compare mode) Skip one backend if it has issues on this cluster.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep batch_size × gpu_concurrency for one backend instead "
             "of comparing both. Defaults to backend=transformers.",
    )
    parser.add_argument(
        "--backend", choices=["transformers", "vllm"], default="transformers",
        help="(sweep mode) Which backend to sweep.",
    )
    parser.add_argument(
        "--sweep-batch-sizes", default="1,2,4",
        help="(sweep mode) Comma-separated batch sizes to try. "
             "Default range stays under T4 OOM ceiling (batch=8 OOMs "
             "for the transformers backend; vLLM's continuous batching "
             "has different VRAM dynamics and may tolerate more).",
    )
    parser.add_argument(
        "--sweep-concurrencies", default="1,2",
        help="(sweep mode) Comma-separated gpu_concurrency values to try.",
    )
    args = parser.parse_args()

    if args.output_root is None:
        # Match run.py's precedence: /mnt/cluster_storage (NFS, visible
        # in the Workspace browser) → $ANYSCALE_ARTIFACT_STORAGE (S3)
        # → /tmp (off-cluster fallback only; not shared, won't work
        # across workers).
        if os.path.isdir("/mnt/cluster_storage"):
            args.output_root = "/mnt/cluster_storage/velvet"
        else:
            args.output_root = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "/tmp")

    if args.sweep:
        # Stretch goal: throughput tuning
        batch_sizes = [int(x) for x in args.sweep_batch_sizes.split(",")]
        concurrencies = [int(x) for x in args.sweep_concurrencies.split(",")]
        results = []
        for bs in batch_sizes:
            for conc in concurrencies:
                stats = run_one(
                    args.backend, args.manifest, args.output_root,
                    args.limit, bs, conc,
                )
                stats["rows"] = count_rows(stats["output"])
                if stats["rows"] > 0 and stats.get("elapsed_sec"):
                    stats["throughput"] = stats["rows"] / stats["elapsed_sec"]
                results.append(stats)
        Path("sweep_results.json").write_text(json.dumps(results, indent=2))
        table_md = render_sweep_table(results)
        Path("sweep_table.md").write_text(table_md)
        log.info("Wrote sweep_results.json and sweep_table.md")
        print("\n" + table_md)
        return

    # Default: compare backends
    backends = ["transformers", "vllm"]
    if args.skip:
        backends = [b for b in backends if b != args.skip]

    results = []
    for backend in backends:
        stats = run_one(
            backend, args.manifest, args.output_root,
            args.limit, args.batch_size, args.gpu_concurrency,
        )
        stats["rows"] = count_rows(stats["output"])
        if stats["rows"] > 0 and stats.get("elapsed_sec"):
            stats["throughput"] = stats["rows"] / stats["elapsed_sec"]
        results.append(stats)

    Path("benchmark_results.json").write_text(json.dumps(results, indent=2))
    table_md = render_table(results)
    Path("benchmark_table.md").write_text(table_md)
    log.info("Wrote benchmark_results.json and benchmark_table.md")
    print("\n" + table_md)


if __name__ == "__main__":
    main()
