"""
Stage 3: Captioning. Two interchangeable implementations.

Backend A — `SmolVLMTransformersActor` + `caption_with_transformers`:
    HuggingFace transformers loaded into a Ray Data actor pool.
    Universal model compatibility, code-level transparency,
    `model.generate()` based generation (no continuous batching).

Backend B — `caption_with_vllm`:
    Ray Data LLM (`vLLMEngineProcessorConfig` + `build_processor`).
    Continuous batching, PagedAttention. Higher steady-state throughput
    but more T4-specific configuration knobs.

Both backends produce identical output schemas, so all downstream code
(write_parquet, schema validation) is unchanged.

A note on the API: `concurrency=` vs `compute=`
-----------------------------------------------
The brief's hint suggests `map_batches(MyClass, concurrency=N, …)`. As
of Ray 2.55, the docstring for Dataset.map_batches says verbatim:
"concurrency – This argument is deprecated. Use compute argument."
We therefore use `compute=ray.data.ActorPoolStrategy(size=N)`, which is
the canonical form in the latest Ray Data docs. Note this is a
Dataset-API change only — `vLLMEngineProcessorConfig` still uses
`concurrency=` as a current parameter (it controls workers for data
parallelism in the LLM processor, a different concept).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import ray
from PIL import Image

from velvet.config import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    NUM_FRAMES,
    PROMPT,
    PipelineConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Backend A: HuggingFace transformers
# ---------------------------------------------------------------------

class SmolVLMTransformersActor:
    """
    Stateful actor: loads SmolVLM2 once in __init__, captions batches
    in __call__.

    Why a class (not a function)
    ----------------------------
    The model is ~1.2 GB and takes ~10s to load. An actor amortizes
    that load across hundreds of batches. A function would reload every
    batch — fatal for throughput.

    Why this is canonical
    ---------------------
    See https://docs.ray.io/en/latest/data/batch_inference.html — the
    exact pattern (class with __init__ + __call__, passed to
    map_batches with ActorPoolStrategy) is the recommended idiom for
    GPU inference.
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        log.info("transformers_actor_loading model=%s", MODEL_NAME)
        t0 = time.time()

        # WHY do_image_splitting=False: SmolVLM2's default image
        # processor crops each frame into multiple sub-tiles (~2x),
        # each producing ~729 patches. With 16 frames that yields
        # ~23K vision tokens — the LLM attention matrix then balloons
        # to ~18 GiB in fp16 and OOMs on T4 even at batch=1 (see
        # runs/smoke_20260511_000124_tf4_b1.log). Disabling splitting
        # halves the token count and brings the attention matrix to
        # ~4.4 GiB, well within T4 budget.
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, do_image_splitting=False
        )
        # fp16 because T4 is Turing — bf16 isn't natively supported
        # (would silently fall back to fp32 and tank throughput).
        # `dtype=` replaces the deprecated `torch_dtype=` in transformers
        # 5.x; the older name still works but emits a warning.
        # device_map="cuda" + low_cpu_mem_usage=True streams weights
        # straight onto the GPU rather than materializing a full CPU
        # copy first. Without this, on a g4dn.xlarge (16 GB CPU RAM)
        # the .from_pretrained → .to('cuda') sequence transiently uses
        # ~2x the model size in CPU RAM, which combined with the
        # Python/torch/vllm process footprint trips the OOM killer
        # and the actor restarts in a loop. Symptom observed:
        # "Worker exit type: SYSTEM_ERROR ... killed by the OOM killer".
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        ).eval()

        log.info("transformers_actor_ready elapsed_sec=%.1f", time.time() - t0)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Caption a batch of videos.

        batch is a Ray Data dict-of-arrays:
            video_id     : (B,) str
            frames       : (B, 16, 384, 384, 3) uint8
            num_frames   : (B,) int
            duration_sec : (B,) float
        """
        import torch

        batch_size = len(batch["video_id"])

        # WHY: pass each video's 16 frames as 16 *images* (not as a single
        # video), mirroring the vLLM backend below. Between transformers
        # 4.45 (when this was originally written) and 4.53.2 (vLLM 0.10's
        # floor) the multimodal processor's `videos=` argument changed
        # in a way that misreads the first dim of any nested input as
        # "video count" — both list-of-list-of-PIL and
        # list-of-ndarray-(T,H,W,C) raise:
        #   "videos [16,16,...] should be the same as text [1,1,...]"
        # The image path (16 images per prompt with `<image>` tokens) is
        # stable across the 4.x and 5.x lines, and at the tensor level
        # produces the same input — SmolVLM2's chat template ultimately
        # expands a `<video>` placeholder into N image tokens internally
        # anyway. Output schema is unchanged.
        # WHY images_per_video is nested (not flat): when batch_size > 1
        # the processor needs to pair each text with its own image list,
        # otherwise it raises "The number of images in the text [16, 16]
        # and images [32] should be the same" — text reports per-item
        # placeholder counts while a flat image list reports a single
        # total (see runs/smoke_transformers_d1.log for the original
        # occurrence). The latent bug was only triggered once batching
        # was actually exercised at batch_size > 1.
        messages_per_video = []
        images_per_video: list[list[Image.Image]] = []
        for i in range(batch_size):
            video_frames = [
                Image.fromarray(batch["frames"][i, j]) for j in range(NUM_FRAMES)
            ]
            images_per_video.append(video_frames)
            content = [{"type": "image"}] * NUM_FRAMES + [
                {"type": "text", "text": PROMPT},
            ]
            messages_per_video.append([{"role": "user", "content": content}])

        text_per_video = [
            self.processor.apply_chat_template(m, add_generation_prompt=True)
            for m in messages_per_video
        ]

        # WHY: in transformers 5.x there's a deprecation warning that
        # claims kwargs should be nested under `processor_kwargs={...}`,
        # but in practice nesting `return_tensors` there silently drops
        # it (the processor returns numpy arrays which then fail at
        # `.to("cuda")` with `AttributeError: 'numpy.ndarray' object has
        # no attribute 'to'`). Pass `return_tensors="pt"` at top level
        # — it's still honored. `padding=True` triggers the deprecation
        # noise (5x warnings per batch); accept it. Revisit when the
        # transformers API stabilises (likely 5.1+).
        inputs = self.processor(
            text=text_per_video,
            images=images_per_video,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        # WHY: BatchFeature.to(device, dtype) would cast EVERY tensor
        # (including int64 input_ids and attention_mask) to fp16, which
        # corrupts the embedding lookup. Cast only the floating-point
        # pixel inputs.
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # inference_mode is strictly faster than no_grad — disables
        # version counter tracking that no_grad still does.
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy → deterministic, fair benchmark
                temperature=None,
                top_p=None,
            )

        # generate() returns input + output concatenated; slice off prompt.
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        captions = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )
        captions = [c.strip() for c in captions]

        return {
            "video_id": batch["video_id"],
            "caption": np.array(captions),
            "num_frames": batch["num_frames"],
            "duration_sec": batch["duration_sec"],
        }


def caption_with_transformers(
    ds: ray.data.Dataset,
    cfg: PipelineConfig,
) -> ray.data.Dataset:
    """
    Wire the transformers actor into the Ray Data pipeline.

    Note on the Ray Data API
    ------------------------
    Recent Ray docs prefer `compute=ray.data.ActorPoolStrategy(size=N)`
    over the older `concurrency=N` kwarg (which is now deprecated).
    The ActorPoolStrategy form is more explicit and supports
    autoscaling pools (m, n) when desired:
        compute=ray.data.ActorPoolStrategy(min_size=2, max_size=4)
    """
    kwargs: dict[str, Any] = {
        "compute": ray.data.ActorPoolStrategy(size=cfg.gpu_concurrency),
        "num_gpus": cfg.num_gpus_per_actor,
        "batch_size": cfg.batch_size,
        # WHY: disable Ray's auto-restart-on-actor-death loop. Without
        # these, an actor that crashes mid-`generate()` (CUDA segfault,
        # OOM, etc.) gets silently respawned, re-downloads the model,
        # crashes again — wasting minutes per cycle and hiding the real
        # error. With max_restarts=0 + max_task_retries=0 the first
        # crash propagates a real traceback to the driver and we abort.
        # Production may want a larger budget (e.g. 2 retries to absorb
        # transient hardware blips); for development & this take-home,
        # fail-fast is the right default.
        # API note: Ray Data's `map_batches` accepts these via
        # `**ray_remote_args` — pass as top-level kwargs, NOT nested
        # under a `ray_remote_args=` dict (which raises
        # `Invalid option keyword ray_remote_args for actors`).
        "max_restarts": 0,
        "max_task_retries": 0,
    }
    if cfg.use_node_tags:
        # Explicit pin to GPU worker nodes, matching the spec's
        # heterogeneous-scheduling requirement.
        kwargs["resources"] = {"gpu_node": 1}
    return ds.map_batches(SmolVLMTransformersActor, **kwargs)


# ---------------------------------------------------------------------
# Backend B: Ray Data LLM with vLLM
# ---------------------------------------------------------------------

def caption_with_vllm(
    ds: ray.data.Dataset,
    cfg: PipelineConfig,
) -> ray.data.Dataset:
    """
    Wire Ray Data LLM with vLLM into the pipeline.

    Why this differs structurally from the transformers path
    --------------------------------------------------------
    vLLM speaks a request shape, not a dict-of-arrays batch. We pass
    each video as a list of 16 PIL Images via `multi_modal_data`.
    The processor handles continuous batching internally — that's
    where the throughput win comes from.

    Note: vLLMEngineProcessorConfig still uses `concurrency=` as a
    current (non-deprecated) parameter. Different concept from the
    deprecated `concurrency=` on Dataset.map_batches.

    T4-specific configuration
    -------------------------
    - dtype="half"        : T4 is Turing; bf16 unsupported.
    - enforce_eager=True  : skip CUDA graph capture (some kernels are
                            unstable on T4 + SmolVLM2).
    - attention_backend="TRITON_ATTN" (engine_kwarg) : vLLM's default
                            FlashAttention requires CC >= 8.0 (Ampere+);
                            T4 is CC 7.5. The historical
                            VLLM_ATTENTION_BACKEND env var was REMOVED
                            from vllm.envs in 0.20.2 (verified: vLLM
                            logs "Unknown vLLM environment variable
                            detected" and ignores the value); the
                            replacement is this engine arg, type
                            AttentionBackendEnum | None. Without it
                            vLLM auto-picks FLASHINFER on T4 and fails
                            to JIT sm_75 kernels (Ninja exit 127). The
                            even older `VLLM_USE_V1=0` escape hatch is
                            also dead. TRITON_ATTN JITs via Triton +
                            gcc (provided by `build-essential` in the
                            Containerfile).
    - max_model_len=4096  : trim KV cache size to fit in 16 GB.
    """
    # WHY: `build_llm_processor` was renamed to `build_processor` in
    # Ray Data LLM (the old name still works but emits a warning that
    # promises a future error). Use the new name directly.
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig

    config = vLLMEngineProcessorConfig(
        model_source=MODEL_NAME,
        task_type="generate",
        engine_kwargs=dict(
            dtype="half",
            enforce_eager=True,
            max_model_len=4096,
            # WHY attention_backend="TRITON_ATTN":
            # vLLM 0.20.2 removed the VLLM_ATTENTION_BACKEND env var
            # (verified: not present in vllm.envs as of 0.20.2; vLLM
            # logs "Unknown vLLM environment variable detected"). The
            # replacement is the `attention_backend` engine arg, taking
            # an AttentionBackendEnum name. Without it, vLLM
            # auto-selects FLASHINFER on Turing (CC 7.5 / T4), then
            # fails at runtime trying to JIT-compile a FlashInfer
            # kernel for sm_75 (Ninja exit 127, see
            # runs/smoke_20260511_000954_vllm_l5_b2_g1.log). TRITON_ATTN
            # is the only backend whose supports_compute_capability
            # returns True unconditionally and JITs via Triton + gcc,
            # both of which are present on the worker image.
            attention_backend="TRITON_ATTN",
            # Tell vLLM how many media items to expect per request.
            # We pass 16 frames as 16 images (most robust path on
            # current vLLM multimodal API).
            limit_mm_per_prompt={"image": NUM_FRAMES},
        ),
        concurrency=cfg.gpu_concurrency,  # vLLM processor's own param
        batch_size=cfg.batch_size,
    )

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        # Convert (16, 384, 384, 3) uint8 to a list of PIL Images.
        frames = row["frames"]
        pil_frames = [Image.fromarray(frames[j]) for j in range(NUM_FRAMES)]

        # 16 image placeholders + 1 text instruction. SmolVLM2's
        # video token expansion isn't universally supported in vLLM
        # video paths yet; passing as 16 images is the robust path.
        content = [{"type": "image"}] * NUM_FRAMES + [
            {"type": "text", "text": PROMPT}
        ]

        return {
            "messages": [{"role": "user", "content": content}],
            "sampling_params": {
                "temperature": 0.0,  # greedy, matches transformers path
                "max_tokens": MAX_NEW_TOKENS,
            },
            "multi_modal_data": {"image": pil_frames},
            # Pass through fields we want in the output.
            "video_id": row["video_id"],
            "num_frames": row["num_frames"],
            "duration_sec": row["duration_sec"],
        }

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "video_id": row["video_id"],
            "caption": row["generated_text"].strip(),
            "num_frames": row["num_frames"],
            "duration_sec": row["duration_sec"],
        }

    processor = build_processor(
        config, preprocess=preprocess, postprocess=postprocess
    )
    return processor(ds)
