# Containerfile for Velvet-Creator
#
# Why a custom image: PyAV needs ffmpeg as a system dep, which can't
# be installed via pure-Python requirements.txt. Anyscale's docs say:
# pure Python -> requirements.txt; system deps -> custom image.
#
# Why this base: Anyscale's Ray image gives us a tested combination of
# CUDA + Python + Ray, avoiding version-mismatch debugging.

FROM anyscale/ray:2.55.1-slim-py313-cu129

# System deps:
#   ffmpeg          — PyAV video decode
#   build-essential — gcc/g++/make for triton JIT in vLLM workers.
#                     Without it, vLLM engine init fails with
#                     "RuntimeError: Failed to find C compiler.
#                     Please specify via CC environment variable
#                     or set triton.knobs.build.impl." (see
#                     runs/smoke_vllm.log:1346 for the original
#                     occurrence).
RUN sudo apt-get update \
 && sudo apt-get install -y --no-install-recommends ffmpeg build-essential \
 && sudo apt-get clean \
 && sudo rm -rf /var/lib/apt/lists/*

# Python deps (both backends + dev tools).
#
# WHY inlined instead of `COPY requirements.txt && pip install -r`:
# Anyscale's hosted image-build service rejects Containerfiles that
# COPY from the local context ("invalid containerfile: COPY from a
# local directory not allowed"). The deps below must stay in sync
# with requirements.txt — see CLAUDE.md "Dependency sync rule".
RUN pip install --no-cache-dir \
      'ray[data]>=2.55.0' \
      'torch>=2.0' \
      av \
      pillow \
      'transformers>=4.53.2' \
      accelerate \
      huggingface_hub \
      num2words \
      'vllm>=0.10.0' \
      'smart_open[s3]' \
      boto3 \
      jupyter \
      ipython

# T4-specific note (no ENV needed any more):
# vLLM's default FlashAttention requires compute capability >= 8.0
# (Ampere+); T4 is Turing/CC 7.5. The historical escape hatch was
# the VLLM_ATTENTION_BACKEND=TRITON_ATTN env var, but vLLM 0.20.2
# REMOVED that variable from vllm.envs (it now logs "Unknown vLLM
# environment variable detected" and ignores the value). The
# current knob is the `attention_backend` engine arg, set in
# src/velvet/captioner.py:caption_with_vllm. We keep build-essential
# above because TRITON_ATTN still JITs via Triton + gcc.
#
# Earlier `VLLM_USE_V1=0` was a yet-older escape hatch; also dead.
