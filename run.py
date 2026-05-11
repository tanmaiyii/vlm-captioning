"""
CLI entrypoint for production runs.

Why this is a thin script (not the main code)
---------------------------------------------
- Keeps argparse out of the library code. `velvet.pipeline.run` is
  importable from notebooks, tests, or other scripts without dragging
  in CLI parsing.
- Matches the layout the Ray batch-inference docs use for production
  examples: a `run.py` at the root, library code in `src/velvet/`.

Usage
-----
    python run.py --manifest manifest.csv --backend transformers
    python run.py --manifest manifest.csv --backend vllm --limit 5
    python run.py --manifest manifest.csv --batch-size 16 --gpu-concurrency 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make `src/` importable when running the file directly.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC)

import ray  # noqa: E402

from velvet.config import PipelineConfig  # noqa: E402
from velvet.pipeline import run  # noqa: E402


def _init_ray() -> None:
    """
    Ship `src/velvet/` to every worker so they can import `velvet.*`.

    The Anyscale workspace propagates pip-installed deps automatically,
    but NOT the workspace's own source tree. Without this, GPU-worker
    actors raise `ModuleNotFoundError: No module named 'velvet'` when
    Ray tries to deserialize the actor class.

    `py_modules` is targeted: it only uploads the velvet package, not
    the whole repo (avoids shipping logs, sample output Parquets,
    cached HF weights, etc).
    """
    velvet_pkg = os.path.join(_SRC, "velvet")
    ray.init(
        runtime_env={
            "py_modules": [velvet_pkg],
            # WHY worker_process_setup_hook: Ray 2.55.1's Ray Data LLM
            # references `vllm.inputs.data.TokensPrompt` at
            # `ray/llm/_internal/batch/stages/vllm_engine_stage.py:538`,
            # but vLLM 0.20.2 removed the `vllm.inputs.data` submodule
            # (the symbol now lives at `vllm.inputs.TokensPrompt`). The
            # shim aliases the old path to the new one on each worker
            # process before Ray hits the call site. Harmless when only
            # the transformers backend is used (silent no-op if vllm
            # isn't importable on the worker).
            "worker_process_setup_hook":
                "velvet._vllm_compat.install_vllm_inputs_data_shim",
            # WHY no VLLM_ATTENTION_BACKEND in env_vars: vLLM 0.20.2
            # removed the variable from vllm.envs ("Unknown vLLM
            # environment variable detected" in worker logs). The T4
            # backend selection now happens via the `attention_backend`
            # engine_kwarg in captioner.caption_with_vllm. Setting the
            # env var here would be a no-op.
            #
            # WHY PYTORCH_CUDA_ALLOC_CONF=expandable_segments: tested
            # during the OOM hunt (see CHANGES.md 2026-05-11). It
            # neither helped nor harmed once do_image_splitting=False
            # was applied to the processor; we leave it set because it
            # cheaply guards against transient fragmentation under
            # tighter batch sizes and matches the PyTorch OOM message's
            # own suggestion.
            "env_vars": {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        },
        ignore_reinit_error=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Velvet-Creator video captioning pipeline.")
    parser.add_argument("--manifest", required=True,
                        help="CSV with column 'video_uri'")
    parser.add_argument("--output", default=None,
                        help="Output dir. Default: /mnt/cluster_storage/velvet/captions-<backend> "
                             "(visible in the Workspace file browser). Falls back to "
                             "$ANYSCALE_ARTIFACT_STORAGE/captions-<backend> if that mount is absent.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N videos (for smoke testing)")
    parser.add_argument("--backend", choices=["transformers", "vllm"],
                        default="transformers")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-concurrency", type=int, default=2)
    parser.add_argument("--use-node-tags", action="store_true",
                        help="Pin decode to nodes tagged 'cpu_node' and the "
                             "GPU stage to 'gpu_node'. Requires the cluster "
                             "to expose those custom resources. See WRITEUP "
                             "§heterogeneous-scheduling.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = PipelineConfig(
        backend=args.backend,
        batch_size=args.batch_size,
        gpu_concurrency=args.gpu_concurrency,
        use_node_tags=args.use_node_tags,
    )

    if args.output is None:
        # WHY default to /mnt/cluster_storage/velvet/ rather than
        # $ANYSCALE_ARTIFACT_STORAGE: both are NFS-equivalent shared
        # storage per the take-home spec, but only /mnt/cluster_storage
        # renders in the Anyscale Workspace file browser — reviewers
        # can eyeball captions without juggling S3 credentials. Fall
        # back to $ANYSCALE_ARTIFACT_STORAGE if cluster_storage isn't
        # mounted (e.g. running this script outside Anyscale).
        if os.path.isdir("/mnt/cluster_storage"):
            args.output = cfg.resolved_output("/mnt/cluster_storage/velvet")
        else:
            artifact = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
            if not artifact:
                parser.error(
                    "Neither /mnt/cluster_storage nor ANYSCALE_ARTIFACT_STORAGE "
                    "is available. Pass --output explicitly."
                )
            args.output = cfg.resolved_output(artifact)

    _init_ray()

    stats = run(
        manifest_path=args.manifest,
        output_path=args.output,
        cfg=cfg,
        limit=args.limit,
    )
    print(f"\nCompleted in {stats['elapsed_sec']:.1f}s. Output: {stats['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
