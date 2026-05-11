"""
Configuration constants.

Centralized so every magic number appears exactly once. Reviewers can
audit decisions in one place; tests can patch values cleanly.
"""

from dataclasses import dataclass, field
from typing import Literal

# --- Model & sampling ------------------------------------------------

MODEL_NAME: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
"""SmolVLM2-500M-Video-Instruct.

Picked because:
- Video-native: takes a stack of frames as one input. Image-only
  alternatives need temporal pooling, which loses temporal grounding.
- Fits T4: ~1.2 GB fp16 weights, leaves headroom on a 16 GB GPU.
- T4-compatible: works with default attention + fp16. No bf16, no FA2.
- vLLM 0.10+ supports it, enabling the dual-backend comparison.
"""

NUM_FRAMES: int = 16
"""Per the take-home spec, confirmed by Ali: 16 evenly-spaced frames
per video, regardless of video duration."""

FRAME_SIZE: int = 384
"""SmolVLM2's vision encoder is trained at 384px. Larger wastes
encode compute (downsampled internally); smaller hurts caption quality."""

PROMPT: str = "Describe what is happening in this video in one sentence."
"""Instruction-tuned models behave better with explicit single-task
prompts. 'in one sentence' caps verbosity and keeps the parquet
output predictable."""

MAX_NEW_TOKENS: int = 64
"""One sentence ~ 30-60 tokens. 64 is comfortable headroom."""


# --- Default cluster shape -------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """All knobs in one place. Override per-run via CLI flags or
    notebook cell — never edit constants for experiments.
    """

    # Backend selection
    backend: Literal["transformers", "vllm"] = "transformers"

    # GPU stage
    gpu_concurrency: int = 2
    """One actor per T4. With 2 T4 worker nodes, concurrency=2 saturates
    them. concurrency=1 leaves a GPU idle; concurrency=4 OOMs."""

    batch_size: int = 2
    """Conservative T4 default. The real OOM ceiling on T4 isn't a
    function of per-item activations — it's the LLM attention matrix
    over the vision-token sequence. With SmolVLM2's default
    do_image_splitting=True, 16 frames generate ~23K vision tokens and
    PyTorch SDPA's math backend materializes an ~18 GiB attention
    matrix even at batch=1 (see WRITEUP §Sizing and CHANGES.md
    2026-05-11). captioner.py disables splitting at processor
    construction; with that, batch=2 fits comfortably (~4.4 GiB
    attention + ~1.2 GiB weights + CUDA overhead). Tune upward (4, 8)
    once a sweep confirms steady-state VRAM headroom."""

    num_gpus_per_actor: float = 1.0
    """Each actor reserves a full GPU. Fractional values (e.g., 0.5)
    would let two actors share a T4 — possible for tiny models but
    we don't need it here."""

    # CPU stages (decode + fetch). None = let Ray Data autoscale,
    # which is the recommendation in the Ray docs for I/O-bound work.
    decode_concurrency: tuple[int, int] | None = None
    fetch_concurrency: tuple[int, int] | None = None

    # Heterogeneous scheduling (the spec's "decode CPU only, caption GPU
    # only" requirement). When True, decode requests `cpu_node` custom
    # resource and the GPU stage requests `gpu_node`. Requires the
    # cluster to expose those resources (Anyscale worker-group config,
    # or `ray start --resources='{"cpu_node": 1}'` locally per the
    # spec's own hint). When False (default), placement falls out from
    # `num_gpus=1` on the GPU stage alone — works on any cluster.
    use_node_tags: bool = False

    # I/O
    output_subdir: str = "captions"

    def resolved_output(self, base: str) -> str:
        """`{base}/captions-{backend}` — joined safely (handles
        s3:// schemes and trailing slashes).

        Production default base in run.py is `/mnt/cluster_storage/velvet`
        (NFS-mounted, visible in the Anyscale Workspace file browser);
        falls back to `$ANYSCALE_ARTIFACT_STORAGE` when cluster_storage
        isn't mounted. Both qualify as "shared storage" per the spec.
        """
        return f"{base.rstrip('/')}/{self.output_subdir}-{self.backend}"


# Sentinel default used when no config is passed.
DEFAULT_CONFIG = PipelineConfig()
