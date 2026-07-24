"""Regression tests for timing.adapt_whisperx_words timing safety.

Asserts every emitted word has finite, strictly positive (end > start),
monotonic, non-overlapping timing — and that the function fails safe
(returns None) on bad alignments rather than manufacturing zero-duration
or out-of-order anchors. Interpolation is reserved ONLY for genuinely
absent timestamps; present-but-invalid = corruption = None.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docker"))
from timing import adapt_whisperx_words  # noqa: E402

adapt = adapt_whisperx_words


def _assert_valid(words, seg_start, seg_end):
    """Every word: finite, strictly positive, monotonic, in-bounds."""
    assert words is not None, "expected words, got None"
    prev_end = seg_start
    for w in words:
        s, e = w["start"], w["end"]
        assert math.isfinite(s) and math.isfinite(e), f"non-finite: {w}"
        assert e > s, f"non-positive duration: {w}"
        assert s >= prev_end, f"non-monotonic: {w} (prev_end={prev_end})"
        assert s >= seg_start and e <= seg_end, f"out of bounds: {w}"
        assert "probability" in w and "word" in w, f"missing keys: {w}"
        prev_end = e


# --- happy paths ---

def test_all_aligned_words_pass_through():
    words = [
        {"word": "hello", "start": 0.1, "end": 0.5, "score": 0.9},
        {"word": " world", "start": 0.6, "end": 1.0, "score": 0.8},
    ]
    out = adapt(words, 0.0, 2.0)
    _assert_valid(out, 0.0, 2.0)
    assert out[0]["probability"] == 0.9  # score -> probability
    assert out[0]["word"] == "hello"


def test_unaligned_words_interpolated_evenly():
    words = [
        {"word": "a", "start": 0.0, "end": 0.2, "score": 0.9},
        {"word": "b", "start": None, "end": None, "score": 0.5},
        {"word": "c", "start": None, "end": None, "score": 0.5},
        {"word": "d", "start": 1.0, "end": 1.2, "score": 0.9},
    ]
    out = adapt(words, 0.0, 2.0)
    _assert_valid(out, 0.0, 2.0)
    assert out[1]["start"] == 0.2 and out[2]["end"] == 1.0


# --- absence vs corruption ---

def test_no_aligned_words_returns_none():
    words = [{"word": "x", "start": None, "end": None, "score": 0.5}]
    assert adapt(words, 0.0, 2.0) is None


def test_empty_returns_none():
    assert adapt([], 0.0, 2.0) is None


def test_one_present_one_absent_returns_none():
    words = [{"word": "x", "start": 0.5, "end": None, "score": 0.9}]
    assert adapt(words, 0.0, 2.0) is None
    words2 = [{"word": "x", "start": None, "end": 0.5, "score": 0.9}]
    assert adapt(words2, 0.0, 2.0) is None


def test_non_numeric_timestamps_returns_none():
    words = [{"word": "x", "start": "abc", "end": 0.5, "score": 0.9}]
    assert adapt(words, 0.0, 2.0) is None


def test_inverted_timestamps_returns_none():
    words = [{"word": "x", "start": 0.5, "end": 0.3, "score": 0.9}]
    assert adapt(words, 0.0, 2.0) is None


def test_out_of_bounds_returns_none():
    words = [{"word": "x", "start": 3.0, "end": 3.5, "score": 0.9}]
    assert adapt(words, 0.0, 2.0) is None


def test_non_finite_score_returns_none():
    words = [{"word": "x", "start": 0.0, "end": 0.5, "score": float("nan")}]
    assert adapt(words, 0.0, 2.0) is None


# --- second-pass / output safety ---

def test_overlap_anchor_fails_safe():
    words = [
        {"word": "a", "start": 0.0, "end": 0.5, "score": 0.9},
        {"word": "b", "start": 0.3, "end": 0.6, "score": 0.9},
    ]
    assert adapt(words, 0.0, 2.0) is None


def test_zero_duration_anchor_returns_none():
    words = [{"word": "a", "start": 0.5, "end": 0.5, "score": 0.9}]
    assert adapt(words, 0.0, 2.0) is None


def test_rounding_collapse_fails_safe():
    words = [{"word": "a", "start": 0.0001, "end": 0.0002, "score": 0.9}]
    assert adapt(words, 0.0, 2.0) is None


def test_word_concatenation_reproduces_segment_text():
    """stable-ts rebuilds segment text by concatenating word strings.
    WhisperX words lack leading spaces; the adapter must add them so
    ''.join(w['word'] for w in out) == original segment text.
    """
    words = [
        {"word": "wildlings", "start": 0.0, "end": 0.4, "score": 0.9},
        {"word": "do", "start": 0.5, "end": 0.6, "score": 0.8},
        {"word": "a", "start": 0.6, "end": 0.7, "score": 0.7},
        {"word": "thing", "start": 0.7, "end": 0.9, "score": 0.8},
        {"word": "like", "start": 0.9, "end": 1.0, "score": 0.8},
        {"word": "this.", "start": 1.0, "end": 1.2, "score": 0.9},
    ]
    out = adapt(words, 0.0, 2.0)
    _assert_valid(out, 0.0, 2.0)
    reconstructed = "".join(w["word"] for w in out)
    assert reconstructed == "wildlings do a thing like this.", (
        f"text reconstruction mismatch: {reconstructed!r}"
    )


def test_words_preserve_existing_leading_spaces():
    """If a word already has a leading space, don't add another."""
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.9},
        {"word": " world", "start": 0.5, "end": 0.9, "score": 0.8},
    ]
    out = adapt(words, 0.0, 2.0)
    reconstructed = "".join(w["word"] for w in out)
    assert reconstructed == "Hello world", f"got: {reconstructed!r}"


# --- build_result_from_whisperx text-match fail-safe ---

from timing import build_result_from_whisperx  # noqa: E402


def test_build_result_ok_when_text_matches():
    aligned = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "hello world.",
         "words": [
             {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
             {"word": " world.", "start": 0.5, "end": 1.0, "score": 0.9},
         ]},
    ]}
    segs, ok = build_result_from_whisperx(aligned)
    assert ok is True
    assert len(segs) == 1
    assert segs[0]["words"][0]["word"] == "hello"
    assert segs[0]["words"][1]["word"] == " world."


def test_build_result_ok_with_spacing_fix():
    """WhisperX words without leading spaces -> adapter adds them -> match."""
    aligned = {"segments": [
        {"start": 0.0, "end": 1.2, "text": "wildlings do a thing like this.",
         "words": [
             {"word": "wildlings", "start": 0.0, "end": 0.4, "score": 0.9},
             {"word": "do", "start": 0.5, "end": 0.6, "score": 0.8},
             {"word": "a", "start": 0.6, "end": 0.7, "score": 0.7},
             {"word": "thing", "start": 0.7, "end": 0.9, "score": 0.8},
             {"word": "like", "start": 0.9, "end": 1.0, "score": 0.8},
             {"word": "this.", "start": 1.0, "end": 1.2, "score": 0.9},
         ]},
    ]}
    segs, ok = build_result_from_whisperx(aligned)
    assert ok is True, "should match with spacing fix"
    reconstructed = "".join(w["word"] for w in segs[0]["words"])
    assert reconstructed == "wildlings do a thing like this."


def test_build_result_fallback_on_text_mismatch():
    """If word concatenation can't reproduce segment text -> ok=False."""
    aligned = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "completely different text here.",
         "words": [
             {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
             {"word": " world", "start": 0.5, "end": 1.0, "score": 0.9},
         ]},
    ]}
    segs, ok = build_result_from_whisperx(aligned)
    assert ok is False
    assert segs == []


def test_build_result_ok_with_contraction():
    """Contractions like I've should reconstruct correctly."""
    aligned = {"segments": [
        {"start": 0.0, "end": 1.5, "text": "I've never seen this.",
         "words": [
             {"word": "I've", "start": 0.0, "end": 0.3, "score": 0.9},
             {"word": "never", "start": 0.4, "end": 0.7, "score": 0.9},
             {"word": "seen", "start": 0.8, "end": 1.1, "score": 0.9},
             {"word": "this.", "start": 1.2, "end": 1.5, "score": 0.9},
         ]},
    ]}
    segs, ok = build_result_from_whisperx(aligned)
    assert ok is True
    reconstructed = "".join(w["word"] for w in segs[0]["words"])
    assert reconstructed == "I've never seen this."


def test_build_result_ok_with_punctuation():
    """Punctuation attached to words should reconstruct correctly."""
    aligned = {"segments": [
        {"start": 0.0, "end": 2.0, "text": "Hello, world! How are you?",
         "words": [
             {"word": "Hello,", "start": 0.0, "end": 0.3, "score": 0.9},
             {"word": "world!", "start": 0.4, "end": 0.7, "score": 0.9},
             {"word": "How", "start": 0.8, "end": 1.0, "score": 0.9},
             {"word": "are", "start": 1.1, "end": 1.3, "score": 0.9},
             {"word": "you?", "start": 1.4, "end": 2.0, "score": 0.9},
         ]},
    ]}
    segs, ok = build_result_from_whisperx(aligned)
    assert ok is True
    reconstructed = "".join(w["word"] for w in segs[0]["words"])
    assert reconstructed == "Hello, world! How are you?"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))