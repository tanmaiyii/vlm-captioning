# Velvet-Creator

Video captioning pipeline built with Ray Data. Two interchangeable GPU
backends — HuggingFace transformers vs Ray Data LLM with vLLM — for
side-by-side comparison.

## Two ways to run

| Mode | When to use | Entrypoint |
|---|---|---|
| **Notebook** — narrative walkthrough | First time. Validating each stage. Demoing to reviewers. | `notebooks/explore.ipynb` |
| **Production** — modular `run.py` | Real benchmark runs. Submission. Production. | `python run.py …` |

Both paths import from the same modules in `src/velvet/`. There's no
copy-paste, no drift. The notebook is a *demo* of the modules, not a
parallel implementation.

## Layout

```
.
├── src/velvet/              # the library (the actual product)
│   ├── __init__.py
│   ├── config.py           # all tunables in one place
│   ├── sampling.py         # pure logic — frame index math
│   ├── decode.py           # Stage 2: video decode
│   ├── captioner.py        # Stage 3: both backends
│   ├── io.py               # Stages 1 & 4: read manifest, write parquet
│   └── pipeline.py         # wires the four stages
├── notebooks/
│   └── explore.ipynb       # narrative walkthrough
├── run.py                   # thin CLI for production
├── benchmark.py             # run both backends, emit comparison
├── build_manifest.py        # CSV manifest from S3 or local dir
├── requirements.txt         # pinned Python deps
├── Containerfile            # cluster image (adds ffmpeg, sets vLLM env)
├── CLAUDE.md               # constraints for Claude Code
├── WRITEUP.md              # design rationale + benchmarks (the rubric)
├── SLIDES_OUTLINE.md       # 30-min presentation outline
└── README.md               # this file
```

## Quickstart

### Prerequisites

- A cluster with at least 2 CPU workers and 2 GPU workers (NVIDIA T4
  or newer). The take-home target shape is documented in `WRITEUP.md`
  §Sizing — minimum is 1 CPU + 1 GPU but heterogeneous scheduling
  needs ≥2 GPU workers to demonstrate "both GPUs used."
- Python 3.13 (matches the `anyscale/ray:2.55.1-slim-py313-cu129`
  base image we use; older 3.10+ likely works but isn't tested).
- `ffmpeg` and `build-essential` system packages installed (the
  `Containerfile` does this; if running locally use `apt install
  ffmpeg build-essential`).
- `/mnt/cluster_storage/` mounted (Anyscale default) or
  `$ANYSCALE_ARTIFACT_STORAGE` set — used for Parquet output.

### Setup

```bash
# 1. Clone the repo
git clone <repo-url> velvet && cd velvet

# 2. Install Python deps (pinned in requirements.txt)
pip install -r requirements.txt

# 3. (Optional, for the notebook) Jupyter kernel
pip install jupyterlab nbclient

# 4. Verify the model is reachable
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')"
```

### Path A — interactive walkthrough (recommended first read)

```bash
jupyter lab notebooks/explore.ipynb
```

7 sections mirror the four pipeline stages. Each validates one piece.
Total run time ~2.5–3 min on warm workers (~5 min cold).

This is the right starting point — see every intermediate dataset
shape before running at scale.

### Path B — production CLI (`run.py`)

```bash
# 1. Build a manifest (one column: video_uri)
python build_manifest.py \
    --source s3://your-bucket/videos/ \
    --out manifest.csv

# 2. Smoke test — 5 videos
python run.py --manifest manifest.csv --limit 5 \
    --batch-size 8 --gpu-concurrency 2 --backend transformers

# 3. Full run (transformers, the validated path)
python run.py --manifest manifest.csv \
    --batch-size 8 --gpu-concurrency 2 --backend transformers

# 4. With metrics capture (recommended — emits .raystatus.log + .metrics.json)
./runs/run_with_metrics.sh main_transformers \
    python run.py --manifest manifest.csv \
    --batch-size 8 --gpu-concurrency 2 --backend transformers
```

**vLLM backend.** `--backend vllm` is wired but currently produces a
known caption-quality regression (every output identical, see
`WRITEUP.md` §"What I'd do differently" #3). Smoke-tests run cleanly;
the full-scale fix is the first item on the next-day list.

**Batch-size sweep** (for reproducing the WRITEUP §Sizing table):

```bash
for b in 2 4 8; do
  ./runs/run_with_metrics.sh sweep_b${b} \
    python run.py --manifest manifest_30.csv \
      --batch-size $b --gpu-concurrency 2 --backend transformers
done
```

## Heterogeneous CPU/GPU placement

The brief specifies a cluster of 2 CPU workers (8 vCPU each) + 2 GPU
workers (1 T4 + 4 vCPU each). Node placement is expressed via Ray
Data's resource hints:

| Stage | Resource | Lands on |
|---|---|---|
| Stage 1: fetch | `num_cpus=1` | CPU workers (most spare CPU) |
| Stage 2: decode | `num_cpus=1` | CPU workers (most spare CPU) |
| Stage 3: caption | `num_gpus=1` | GPU workers (only nodes with GPUs) |
| Stage 4: write | default CPU | CPU workers |

By default this is sufficient: only GPU workers have GPUs, so the
caption actors place there exclusively. Decode tasks land on whichever
node has free CPU, which is overwhelmingly the CPU workers because the
GPU workers' 4 vCPU are mostly used by the captioner's preprocessing.

For hard guarantees (e.g., if your dashboard shows decode tasks
wandering onto GPU workers), pass `--use-node-tags`:

```bash
python run.py --manifest manifest.csv --use-node-tags ...
```

This adds explicit `resources={"cpu_node": 1}` to decode and
`resources={"gpu_node": 1}` to the GPU stage. Requires the cluster to
expose those custom resources — on Anyscale, configure them in the
worker-group definition; locally, pass them to `ray start` per the
spec's own hint:

```bash
ray start --head --resources='{"cpu_node": 1}'
ray start --address=... --resources='{"gpu_node": 1}'
```

## Configuration

All tunable parameters live in `src/velvet/config.py`. Don't edit
constants for one-off experiments — pass overrides via CLI flags.

| Flag | Default | Notes |
|---|---|---|
| `--manifest` | required | CSV with column `video_uri` |
| `--backend` | `transformers` | `transformers` \| `vllm` |
| `--output` | `/mnt/cluster_storage/velvet/captions-<backend>` (falls back to `$ANYSCALE_ARTIFACT_STORAGE/captions-<backend>` off-workspace) | Parquet output dir |
| `--limit` | None | Cap input rows (smoke tests) |
| `--batch-size` | 2 | GPU batch size. **For benchmark/production, use `--batch-size 8`** (sweep winner; see WRITEUP §Sizing). 2 stays the default for safety on untested clusters. |
| `--gpu-concurrency` | 2 | Number of GPU actors / vLLM workers |
| `--use-node-tags` | False | Pin decode to `cpu_node`, GPU stage to `gpu_node` (custom resources). Cluster must expose them. |

## Output schema

| column | type | example |
|---|---|---|
| `video_id` | string | `"abc123"` |
| `caption` | string | `"A person walks across a kitchen and pours coffee."` |
| `num_frames` | int64 | `300` (source video's total — not 16) |
| `duration_sec` | double | `10.0` |

Identical between backends.

## Committed output (deliverable)

Per the take-home spec ("a sample of the output: the first ~100 rows
of your captions Parquet, committed alongside the code so we can
eyeball quality without re-running"), the head of the transformers
run is checked into the repo:

```
output/
└── captions-transformers/
    ├── preview.csv     # first 100 rows, browsable
    ├── preview.json    # same 100 rows in JSON for IDE pretty-print
    └── <shard>.parquet # the canonical first parquet shard
```

**No `output/captions-vllm/`** — the vLLM full run was deferred this
round (every caption came out identical, suspected `max_model_len=4096`
truncating image tokens before the LLM saw them). The vLLM code path
is still in the repo (`src/velvet/captioner.py:caption_with_vllm`,
notebook §4b, `--backend vllm` CLI flag); WRITEUP §"What I'd do
differently" #3 documents the fix sketch.

The full Parquet from the transformers run lives at
`/mnt/cluster_storage/velvet/captions-transformers/` (NFS, visible in
the Anyscale Workspace file browser without S3 credentials). If
`/mnt/cluster_storage` isn't mounted, it falls back to
`$ANYSCALE_ARTIFACT_STORAGE/captions-transformers/` (S3). Both are
durable across cluster restarts and readable from every worker. The
in-repo files under `output/` are 100-row samples taken from the run
that produced the numbers in `WRITEUP.md` §Observed throughput.

## Key design decisions

The full design narrative is in **`WRITEUP.md`** — the primary
deliverable per the take-home rubric. Every non-obvious choice has
a `# WHY:` comment in the source. Headlines:

- Stages 1, 2, and 4 are CPU; Stage 3 is GPU (heterogeneous scheduling).
- Stage 2 is a stateless function (no expensive setup); Stage 3 is a
  stateful actor class (model loaded once per actor).
- We use `compute=ray.data.ActorPoolStrategy(size=N)` (current API)
  rather than the deprecated `concurrency=N` kwarg.
- 16 frames sampled evenly per video, computed up front so we only
  decode what we keep.
- No `.materialize()` between stages in production — let the streaming
  executor do its job.
- Built BOTH backends behind a `--backend` flag because the brief
  explicitly allowed either approach; building both lets the choice
  be defended with measurements rather than opinion.
