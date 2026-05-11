"""
Pure frame-sampling logic.

Deliberately kept zero-dependency (no Ray, no PyAV, no PIL) so it can
be unit-tested without a GPU cluster. The decode module imports from
here.

This is the *only* module in the project with unit tests, because:
- The sampling formula is the most easily-bug-prone bit of pure logic
  in the pipeline.
- It has no external dependencies, so tests run in milliseconds.
- Every other module either calls Ray (integration territory) or
  loads a model (too heavy for unit tests).

Tests for the rest of the pipeline live as smoke tests in the
notebook and the `--limit 3` CLI invocation.
"""

from typing import Sequence


def evenly_spaced_indices(total_frames: int, num_samples: int) -> list[int]:
    """
    Return `num_samples` integer indices spread evenly across
    [0, total_frames).

    Used by the decode stage to pick which frames to keep from a video.
    Per Ali's confirmation: "16 evenly spaced frames regardless of
    video length."

    Parameters
    ----------
    total_frames : number of frames in the source video. Must be > 0.
    num_samples  : number of frames to keep. Typically 16.

    Returns
    -------
    A list of `num_samples` integers, sorted ascending, all in
    [0, total_frames). May contain duplicates if total_frames < num_samples
    (caller is responsible for handling that edge case — typically by
    repeating the last frame).

    Why this formula
    ----------------
    `round(i * N / num_samples)` for i in [0, num_samples) gives a
    uniform spread that includes the start and lands close to the end
    without falling off. `min(...)` clamps to N-1 for the rare case
    where round() rounds up past the end.

    Alternative considered: `int(i * N / num_samples)` — but that biases
    earlier and misses the last sixth of the video for small num_samples.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    last = max(total_frames - 1, 0)
    return [
        min(round(i * total_frames / num_samples), last)
        for i in range(num_samples)
    ]


def pad_to_length(items: Sequence, target: int) -> list:
    """
    Repeat the last item until len == target. Used to handle videos
    shorter than NUM_FRAMES (rare but real in public datasets).

    >>> pad_to_length([1, 2, 3], 5)
    [1, 2, 3, 3, 3]
    >>> pad_to_length([7], 3)
    [7, 7, 7]
    >>> pad_to_length([1, 2, 3], 2)
    [1, 2, 3]
    """
    if not items:
        raise ValueError("Cannot pad empty sequence")
    if len(items) >= target:
        return list(items)
    last = items[-1]
    return list(items) + [last] * (target - len(items))
