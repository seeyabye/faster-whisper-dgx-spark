"""Unit tests for the Parakeet TDT token-to-word conversion."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docker"))
from parakeet import tokens_to_words, words_to_segments


def test_tokens_to_words_basic():
    tokens = [
        {"token": "W", "start": 0.24, "end": 0.32},
        {"token": "ild", "start": 0.32, "end": 0.48},
        {"token": "l", "start": 0.48, "end": 0.56},
        {"token": "ings", "start": 0.56, "end": 0.72},
        {"token": " do", "start": 0.72, "end": 0.88},
        {"token": " a", "start": 0.88, "end": 0.96},
        {"token": " th", "start": 0.96, "end": 1.04},
        {"token": "ing", "start": 1.04, "end": 1.2},
        {"token": " like", "start": 1.2, "end": 1.44},
        {"token": " this", "start": 1.44, "end": 1.6},
        {"token": ".", "start": 1.6, "end": 1.6},
    ]
    words = tokens_to_words(tokens)
    assert len(words) == 6
    assert words[0]["word"] == "Wildlings"
    assert words[0]["start"] == 0.24
    assert words[0]["end"] == 0.72
    assert words[1]["word"] == " do"
    assert words[5]["word"] == " this."
    assert words[5]["end"] == 1.6


def test_tokens_to_words_zero_duration_punctuation():
    tokens = [
        {"token": "Hello", "start": 0.0, "end": 0.5},
        {"token": " world", "start": 0.5, "end": 1.0},
        {"token": ".", "start": 1.0, "end": 1.0},
        {"token": " How", "start": 1.5, "end": 1.8},
        {"token": "?", "start": 1.8, "end": 1.8},
    ]
    words = tokens_to_words(tokens)
    assert len(words) == 3
    assert words[0]["word"] == "Hello"
    assert words[1]["word"] == " world."
    assert words[1]["end"] == 1.0
    assert words[2]["word"] == " How?"
    assert words[2]["end"] == 1.8


def test_tokens_to_words_empty():
    assert tokens_to_words([]) == []


def test_tokens_to_words_single():
    tokens = [{"token": "Hello", "start": 0.0, "end": 0.5}]
    words = tokens_to_words(tokens)
    assert len(words) == 1
    assert words[0]["word"] == "Hello"


def test_tokens_to_words_skip_empty():
    tokens = [
        {"token": "", "start": 0.0, "end": 0.0},
        {"token": "Hello", "start": 0.0, "end": 0.5},
        {"token": "", "start": 0.5, "end": 0.5},
    ]
    words = tokens_to_words(tokens)
    assert len(words) == 1
    assert words[0]["word"] == "Hello"


def test_words_to_segments_sentence_split():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 1.0},
        {"word": " world.", "start": 0.5, "end": 1.0, "probability": 1.0},
        {"word": " How", "start": 1.5, "end": 1.8, "probability": 1.0},
        {"word": " are", "start": 1.8, "end": 2.0, "probability": 1.0},
        {"word": " you?", "start": 2.0, "end": 2.5, "probability": 1.0},
    ]
    segs = words_to_segments(words)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world."
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.0
    assert len(segs[0]["words"]) == 2
    assert segs[1]["text"] == "How are you?"
    assert segs[1]["start"] == 1.5
    assert segs[1]["end"] == 2.5
    assert len(segs[1]["words"]) == 3


def test_words_to_segments_no_punctuation():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 1.0},
        {"word": " world", "start": 0.5, "end": 1.0, "probability": 1.0},
    ]
    segs = words_to_segments(words)
    assert len(segs) == 1
    assert segs[0]["text"] == "Hello world"


def test_words_to_segments_empty():
    assert words_to_segments([]) == []


def test_text_reconstruction_exact():
    tokens = [
        {"token": "Wildlings", "start": 0.24, "end": 0.72},
        {"token": " do", "start": 0.72, "end": 0.88},
        {"token": " a", "start": 0.88, "end": 0.96},
        {"token": " th", "start": 0.96, "end": 1.04},
        {"token": "ing", "start": 1.04, "end": 1.2},
        {"token": " like", "start": 1.2, "end": 1.44},
        {"token": " this", "start": 1.44, "end": 1.6},
        {"token": ".", "start": 1.6, "end": 1.6},
    ]
    words = tokens_to_words(tokens)
    concat = "".join(w["word"] for w in words)
    assert concat == "Wildlings do a thing like this."


def test_segment_first_word_no_leading_space():
    """First word of each segment must not have a leading space."""
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 1.0},
        {"word": " world.", "start": 0.5, "end": 1.0, "probability": 1.0},
        {"word": " How", "start": 1.5, "end": 1.8, "probability": 1.0},
        {"word": " are", "start": 1.8, "end": 2.0, "probability": 1.0},
        {"word": " you?", "start": 2.0, "end": 2.5, "probability": 1.0},
    ]
    segs = words_to_segments(words)
    for seg in segs:
        first_word = seg["words"][0]["word"]
        assert not first_word.startswith(" "), f"first word has leading space: {first_word!r}"
        # text reconstruction must hold
        concat = "".join(w["word"] for w in seg["words"])
        assert concat == seg["text"], f"mismatch: {concat!r} vs {seg['text']!r}"


def test_pause_based_segment_split():
    """When the gap between words exceeds pause_threshold (0.5s),
    split into a new segment even without punctuation."""
    words = [
        {"word": "But", "start": 2.64, "end": 2.96, "probability": 1.0},
        {"word": " how", "start": 2.96, "end": 3.28, "probability": 1.0},
        # 1.28s gap (no punctuation, but speaker change)
        {"word": " Thoros", "start": 4.56, "end": 5.28, "probability": 1.0},
        {"word": "?", "start": 5.28, "end": 5.28, "probability": 1.0},
    ]
    segs = words_to_segments(words, pause_threshold=0.5)
    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"
    assert segs[0]["text"] == "But how"
    assert segs[1]["text"] == "Thoros?"
    # Verify text reconstruction
    for seg in segs:
        concat = "".join(w["word"] for w in seg["words"])
        assert concat == seg["text"], f"mismatch: {concat!r} vs {seg['text']!r}"


def test_pause_below_threshold_no_split():
    """Gaps below threshold should NOT trigger a split."""
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 1.0},
        {"word": " world", "start": 0.7, "end": 1.0, "probability": 1.0},  # 0.2s gap
    ]
    segs = words_to_segments(words, pause_threshold=0.5)
    assert len(segs) == 1, f"expected 1 segment, got {len(segs)}"


def test_pause_and_punctuation_combined():
    """Both pause and punctuation should create segment boundaries."""
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 1.0},
        {"word": " world.", "start": 0.7, "end": 1.0, "probability": 1.0},
        # 1.0s gap + new segment
        {"word": " How", "start": 2.0, "end": 2.3, "probability": 1.0},
        {"word": " are", "start": 2.3, "end": 2.5, "probability": 1.0},
        {"word": " you?", "start": 2.5, "end": 3.0, "probability": 1.0},
    ]
    segs = words_to_segments(words, pause_threshold=0.5)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world."
    assert segs[1]["text"] == "How are you?"


def test_pause_catch_missing_punctuation():
    """Model drops period between speakers — pause-based split fixes it."""
    # "You can't" + "Enough!" merged because model dropped the period
    words = [
        {"word": "You", "start": 6.72, "end": 6.80, "probability": 1.0},
        {"word": " can", "start": 6.80, "end": 6.88, "probability": 1.0},
        {"word": "'", "start": 6.88, "end": 6.88, "probability": 1.0},
        {"word": "t", "start": 6.96, "end": 6.96, "probability": 1.0},
        # 0.32s gap — below default 0.5s threshold, use 0.3s
        {"word": " en", "start": 7.28, "end": 7.52, "probability": 1.0},
        {"word": "ough", "start": 7.52, "end": 7.76, "probability": 1.0},
        {"word": "!", "start": 7.76, "end": 7.76, "probability": 1.0},
    ]
    segs = words_to_segments(words, pause_threshold=0.3)
    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"
    assert segs[0]["text"] == "You can't"
    assert segs[1]["text"] == "enough!"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))