"""
Compatibility shim: Ray 2.55.1 + vLLM 0.20.2.

Why this exists
---------------
Ray Data LLM (`ray/llm/_internal/batch/stages/vllm_engine_stage.py:538`)
calls `vllm.inputs.data.TokensPrompt(...)`. In vLLM 0.20.2 the `data`
submodule was removed and `TokensPrompt` was hoisted to `vllm.inputs`
directly (verified: `from vllm.inputs import TokensPrompt` succeeds;
`from vllm.inputs.data import TokensPrompt` raises ModuleNotFoundError).

We can't change Ray's vendored code, and we don't want to downgrade
vLLM (it'd cascade into Ray's other vLLM imports). The minimal fix is
to install a synthetic `vllm.inputs.data` module on each worker before
Ray's batch stage hits that line.

Hook this in via `runtime_env.worker_process_setup_hook` —
"velvet._vllm_compat.install_vllm_inputs_data_shim" — so it runs once
per worker process, before any user code or the Ray actor's
`_generate_async` is called.
"""

from __future__ import annotations


def install_vllm_inputs_data_shim() -> None:
    """Install `vllm.inputs.data` as an alias for `vllm.inputs`.

    Idempotent: skips work if `vllm.inputs.data` already exists.
    Silent no-op if vLLM isn't importable (so the shim doesn't break
    transformers-only workers).
    """
    import sys
    import types

    try:
        import vllm.inputs as _vi
    except ImportError:
        return

    if hasattr(_vi, "data"):
        return

    data_mod = types.ModuleType("vllm.inputs.data")
    # Re-export the names Ray's stage uses. TokensPrompt is the only
    # one observed in practice, but mirroring the public surface is
    # cheap and future-proofs against the next Ray-side caller.
    for name in (
        "TokensPrompt",
        "TextPrompt",
        "EmbedsPrompt",
        "PromptType",
        "SingletonPrompt",
    ):
        if hasattr(_vi, name):
            setattr(data_mod, name, getattr(_vi, name))

    _vi.data = data_mod
    sys.modules["vllm.inputs.data"] = data_mod
