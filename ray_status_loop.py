"""
Side-car: poll `ray status` periodically and append timestamped output.

Why this script: the brief asks for "a `ray status` snapshot or a Ray
Dashboard screenshot". Running `ray status` once at peak misses the
warmup and tail. This loop captures the entire run, so the WRITEUP
can show the cluster filling up, both T4s active in steady state,
and the drain at the end.

Usage:
    # Terminal A — start the side-car *before* launching the run:
    python ray_status_loop.py --interval 10 --out runs/ray_status_full_run.log
    # Terminal B — kick off the actual pipeline:
    python run.py --manifest manifest.csv --backend transformers \\
        > runs/full_transformers.log 2>&1
    # When Terminal B exits, hit Ctrl+C in Terminal A.

Notes:
- Pure stdlib + subprocess; no Ray deps imported into this Python
  process so the loop never contends with the pipeline's runtime env.
- Output is plain text, one `ray status` block per tick, separated by
  a header line with the timestamp. Easy to grep ("0/2 GPUs idle"),
  easy to paste verbatim into the WRITEUP.
"""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import subprocess
import sys
import time
from pathlib import Path

_running = True


def _stop(_signum: int, _frame) -> None:
    global _running
    _running = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10,
                        help="Seconds between snapshots.")
    parser.add_argument("--out", default="runs/ray_status.log",
                        help="Append-mode log file.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"Polling ray status every {args.interval}s -> {out_path}")
    print("Ctrl+C to stop.")

    with open(out_path, "a") as f:
        while _running:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n===== {ts} =====\n")
            try:
                result = subprocess.run(
                    ["ray", "status"],
                    capture_output=True, text=True, timeout=8,
                )
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- stderr ---\n")
                    f.write(result.stderr)
            except subprocess.TimeoutExpired:
                f.write("ray status timed out (>8s)\n")
            except FileNotFoundError:
                f.write("ray CLI not on PATH\n")
                f.flush()
                return 1
            f.flush()

            # Sleep in 1s slices so SIGINT is responsive.
            for _ in range(args.interval):
                if not _running:
                    break
                time.sleep(1)

    print("\nStopped. Snapshots written to", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
