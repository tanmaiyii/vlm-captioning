"""
Tests for `velvet.sampling` — the only pure logic in the project worth
unit-testing.

Why we test this and nothing else
---------------------------------
- The sampling formula is the most easily-bug-prone bit of pure logic.
- It has zero external dependencies — tests run in milliseconds.
- A bug here means the model sees the wrong frames, which is silent
  (output schema is unchanged) and would cost real time to diagnose
  without tests.

Why we DON'T test the rest
--------------------------
- `decode.py` requires PyAV + a real video file. Setting up fixtures
  for that is fine but adds dependencies. The notebook + `--limit 3`
  CLI run is a faster integration test for take-home scope.
- `captioner.py` requires SmolVLM2 weights + a GPU. Mocking these
  produces tests that pass even when the real path is broken — they'd
  give false confidence.
- `pipeline.py` and `io.py` are thin wiring; testing them requires a
  Ray cluster, which is the wrong level of test for take-home scope.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest

from velvet.sampling import evenly_spaced_indices, pad_to_length


class TestEvenlySpacedIndices:
    def test_length_matches_num_samples(self) -> None:
        """Always returns exactly num_samples indices, regardless of total."""
        for total in [16, 100, 1000]:
            assert len(evenly_spaced_indices(total, 16)) == 16

    def test_first_index_is_zero(self) -> None:
        """The first sample should always be the start of the video."""
        assert evenly_spaced_indices(100, 16)[0] == 0

    def test_last_index_in_range(self) -> None:
        """No index should exceed total_frames - 1."""
        for total in [16, 100, 1000]:
            assert max(evenly_spaced_indices(total, 16)) <= total - 1

    def test_indices_are_sorted_ascending(self) -> None:
        """We rely on this for in-order linear decode."""
        idx = evenly_spaced_indices(1000, 16)
        assert idx == sorted(idx)

    def test_short_video_clamps_to_last_frame(self) -> None:
        """Total < num_samples means duplicates near the end are OK —
        decode handles padding."""
        idx = evenly_spaced_indices(5, 16)
        assert len(idx) == 16
        assert max(idx) <= 4

    def test_exactly_num_samples_returns_distinct(self) -> None:
        """If total == num_samples, every index should be distinct."""
        idx = evenly_spaced_indices(16, 16)
        assert len(set(idx)) == 16

    def test_zero_total_raises(self) -> None:
        with pytest.raises(ValueError):
            evenly_spaced_indices(0, 16)

    def test_zero_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            evenly_spaced_indices(100, 0)

    def test_evenly_distributed(self) -> None:
        """Indices should be roughly evenly distributed across [0, N).
        For total=1600, num=16, gaps should be ~100 ± 1."""
        idx = evenly_spaced_indices(1600, 16)
        gaps = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
        assert all(99 <= g <= 101 for g in gaps), f"Uneven gaps: {gaps}"


class TestPadToLength:
    def test_pads_with_last_item(self) -> None:
        assert pad_to_length([1, 2, 3], 5) == [1, 2, 3, 3, 3]

    def test_no_pad_when_already_long_enough(self) -> None:
        assert pad_to_length([1, 2, 3], 3) == [1, 2, 3]

    def test_no_truncation(self) -> None:
        """Passing target < len(items) returns items unchanged."""
        assert pad_to_length([1, 2, 3], 2) == [1, 2, 3]

    def test_single_item_pads_correctly(self) -> None:
        assert pad_to_length([7], 3) == [7, 7, 7]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            pad_to_length([], 3)
