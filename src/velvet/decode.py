"""
Stage 2: Video decode + frame sampling.

CPU stage. Stateless function (Ray Data uses Ray tasks for it).
flat_map (not map) so we can return [] on decode failure and drop
corrupt videos cleanly without a separate filter pass.

Why stateless function instead of an actor class
-------------------------------------------------
- No expensive state to amortize. PyAV containers are cheap to open;
  ffmpeg is already in-process; opening a container is microseconds.
- Stateless => Ray tasks (lower scheduling overhead than actors,
  better autoscaling).
- The Ray Data docs explicitly recommend functions for stateless work,
  classes only when __init__ does expensive setup (model loading).
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import av
import numpy as np
from PIL import Image

from velvet.config import FRAME_SIZE, NUM_FRAMES
from velvet.sampling import evenly_spaced_indices, pad_to_length

log = logging.getLogger(__name__)


def decode_and_sample(row: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Decode one video, return one row carrying its 16 sampled frames.

    Parameters
    ----------
    row : dict with keys "bytes" (mp4 bytes) and "path" (str)

    Returns
    -------
    A list with zero or one dict. Zero on decode failure; one on success.
    The single output row has keys:
        video_id     : str   — derived from filename
        frames       : ndarray (16, 384, 384, 3) uint8
        num_frames   : int   — total frames in the SOURCE video
        duration_sec : float — duration of the source video

    Why flat_map (returning a list) instead of map
    -----------------------------------------------
    flat_map lets us emit zero rows on decode failure. With map we'd
    have to emit a sentinel and filter later — extra plumbing.

    Fault tolerance (one of the brief's stretch goals)
    --------------------------------------------------
    Corrupt or unreadable videos return []. They get logged at WARNING
    level and dropped from the dataset rather than killing the
    pipeline. Production would extend this with a dead-letter parquet
    of (video_id, error_type) for offline inspection.
    """
    video_bytes = row["bytes"]
    path = row["path"]
    video_id = path.split("/")[-1].rsplit(".", 1)[0]

    try:
        total_frames, duration_sec = _probe(video_bytes, video_id)
        if total_frames <= 0:
            log.warning("no_frames video_id=%s", video_id)
            return []

        frames = _decode_target_frames(video_bytes, total_frames, video_id)
        if frames is None:
            return []
    except Exception as e:
        log.warning("decode_failed video_id=%s err=%s", video_id, e)
        return []

    return [{
        "video_id": video_id,
        "frames": frames,
        "num_frames": int(total_frames),
        "duration_sec": float(duration_sec),
    }]


def _probe(video_bytes: bytes, video_id: str) -> tuple[int, float]:
    """
    Probe a video for frame count and duration without decoding pixels.

    Most MP4 headers report frame count directly. When they don't (rare
    but happens on some recoded clips), fall back to counting via a
    light demux pass — much cheaper than a full pixel decode because
    we only iterate packets, not frames.
    """
    container = av.open(BytesIO(video_bytes))
    try:
        stream = container.streams.video[0]
        duration_sec = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else 0.0
        )
        total = stream.frames
        if total > 0:
            return total, duration_sec

        # Fallback: count by demuxing packets. Doesn't decode frames, so
        # it's fast (~milliseconds for a typical clip).
        log.info("probing_via_demux video_id=%s", video_id)
        total = sum(1 for pkt in container.demux(video=0) if pkt.pts is not None)
        return total, duration_sec
    finally:
        container.close()


def _decode_target_frames(
    video_bytes: bytes,
    total_frames: int,
    video_id: str,
) -> np.ndarray | None:
    """Linear-decode, keeping only the 16 target frames."""
    if total_frames < NUM_FRAMES:
        log.info("short_video video_id=%s frames=%d", video_id, total_frames)

    target_indices = evenly_spaced_indices(total_frames, NUM_FRAMES)
    target_set = set(target_indices)

    container = av.open(BytesIO(video_bytes))
    try:
        stream = container.streams.video[0]
        # PyAV releases the GIL during threaded decode → multi-core
        # decode without manual multiprocessing.
        stream.thread_type = "AUTO"

        # Linear decode, not seek-per-frame: MP4 seek is keyframe-aligned,
        # so arbitrary seeks re-decode from previous keyframe anyway.
        # Linear is simpler and predictable.
        frames_by_idx: dict[int, np.ndarray] = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in target_set:
                img = frame.to_image()  # PIL Image, RGB
                img = img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
                frames_by_idx[i] = np.asarray(img, dtype=np.uint8)
            if len(frames_by_idx) == NUM_FRAMES:
                break
    finally:
        container.close()

    if not frames_by_idx:
        log.warning("no_frames_decoded video_id=%s", video_id)
        return None

    # Reorder into target_indices' order, padding short videos.
    ordered = [frames_by_idx[i] for i in target_indices if i in frames_by_idx]
    ordered = pad_to_length(ordered, NUM_FRAMES)

    return np.stack(ordered, axis=0)  # (16, 384, 384, 3) uint8
