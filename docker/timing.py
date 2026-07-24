"""Pure timing-adaptation helpers for the whisper-fw server.

Kept in a separate module so they can be unit-tested without importing
faster_whisper/whisperx/fastapi (which require GPU + heavy deps).

These functions convert word/segment objects from faster-whisper and
WhisperX into the stable-ts-compatible schema:
  segments[*].words[*].{word, start, end, probability}

All are timing-safe: invalid inputs return None so the caller can trigger
a whole-request faster-whisper fallback rather than emit bad timings.
"""
import math


def seconds_to_srt_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def faster_whisper_words_to_stable_ts(words):
    """Convert faster-whisper Word objects to stable-ts schema.

    faster-whisper already emits {word, start, end, probability} with the
    exact key stable-ts expects, so this is a straight projection.
    """
    return [
        {
            "word": w.word,
            "start": round(float(w.start), 3),
            "end": round(float(w.end), 3),
            "probability": round(float(w.probability), 3),
        }
        for w in (words or [])
    ]


def adapt_whisperx_words(words, seg_start, seg_end):
    """Adapt WhisperX word objects to the stable-ts schema, timing-safe.

    WhisperX emits {word, start, end, score}. stable-ts expects
    {word, start, end, probability}. Words WhisperX could not align lack
    start/end — allocate each contiguous missing-word run evenly between
    validated anchor timestamps, enforce seg_start<=start<=end<=seg_end
    and monotonic non-overlap. If anchors are inconsistent or the segment
    has no usable aligned words, return None so the caller can fall back.

    Interpolation is reserved ONLY for genuinely absent start/end. Present
    but invalid timestamps (non-numeric, inverted, out of segment, or
    non-finite score) are treated as corruption -> return None immediately.
    """
    if not words:
        return None

    anchored = []
    has_any_aligned = False
    for w in words:
        word_text = w.get("word", "")
        if not word_text:
            continue
        s = w.get("start")
        e = w.get("end")
        score = w.get("score", 0.5)
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score_f):
            return None
        score_r = round(score_f, 3)
        if s is not None and e is not None:
            try:
                s_f, e_f = float(s), float(e)
            except (TypeError, ValueError):
                return None
            if not (math.isfinite(s_f) and math.isfinite(e_f)):
                return None
            if s_f >= e_f or s_f < seg_start or e_f > seg_end:
                return None
            anchored.append({"word": word_text, "start": s_f, "end": e_f,
                             "probability": score_r})
            has_any_aligned = True
        elif s is None and e is None:
            anchored.append({"word": word_text, "start": None, "end": None,
                             "probability": score_r})
        else:
            return None

    if not has_any_aligned:
        return None

    n = len(anchored)
    last_end = seg_start
    i = 0
    while i < n:
        a = anchored[i]
        if a["start"] is not None:
            if a["start"] < last_end:
                return None
            last_end = a["end"]
            i += 1
            continue
        next_start = seg_end
        j = i + 1
        while j < n:
            ns = anchored[j].get("start")
            if ns is not None:
                next_start = max(last_end, ns)
                break
            j += 1
        run_end = i
        while run_end < n and anchored[run_end]["start"] is None:
            run_end += 1
        run_len = run_end - i
        gap = max(0.0, next_start - last_end)
        step = gap / run_len if run_len > 0 else 0.0
        for k in range(run_len):
            a = anchored[i + k]
            s = last_end + step * k
            e = last_end + step * (k + 1)
            a["start"] = round(s, 3)
            a["end"] = round(e, 3)
        last_end = last_end + step * run_len
        i = run_end

    # Reconstruct word text with leading spaces so that concatenation
    # reproduces the segment text (stable-ts rebuilds segment text from
    # words, ignoring the original segment.text field). faster-whisper
    # words carry leading spaces; WhisperX words do not, so we add them.
    # First word keeps its original form; subsequent words get a leading
    # space if they don't already have one.
    out = []
    for idx, a in enumerate(anchored):
        s = max(seg_start, min(seg_end, a["start"]))
        e = max(s, min(seg_end, a["end"]))
        s_r, e_r = round(s, 3), round(e, 3)
        if e_r <= s_r:
            return None
        word_text = a["word"]
        if idx > 0 and not word_text.startswith(" "):
            word_text = " " + word_text
        out.append({"word": word_text, "start": s_r, "end": e_r,
                    "probability": a["probability"]})
    return out if out else None


def _synthesize_words_from_text(text, seg_start, seg_end):
    """Generate a single synthetic word spanning the segment for failed
    alignment. stable-ts DROPS segments that lack a non-empty words
    array, so we must provide one. Using a SINGLE word spanning the full
    segment bounds avoids inventing fake intra-segment timing boundaries
    that stable-ts could split on — the segment passes through intact with
    correct text and WhisperX's own timestamps.

    probability=0.0 marks this as synthetic, not acoustically aligned.
    """
    if not text.strip():
        return []
    s_r = round(seg_start, 3)
    e_r = round(seg_end, 3)
    # Fail safe if rounding collapses to zero/negative duration.
    # The caller (build_result_from_whisperx) treats an empty result for
    # a text-bearing segment as corruption and triggers whole-request
    # faster-whisper fallback (no silent dialogue loss).
    if e_r <= s_r:
        return []
    return [{
        "word": text,
        "start": s_r,
        "end": e_r,
        "probability": 0.0,
    }]


def _text_matches(words_text, seg_text):
    """Check whether concatenated word strings exactly reproduce the segment text.

    stable-ts rebuilds segment text from words[*].word by literal
    concatenation, ignoring the original segment.text field. If they
    diverge by even a space, the SRT output text will differ, so we
    require exact equality. Normalization is intentionally NOT applied
    here — if the spacing heuristic can't reproduce the exact text, the
    whole request falls back to faster-whisper words.
    """
    return words_text == seg_text


def build_result_from_whisperx(aligned_segments):
    """Build stable-ts-compatible result_segments from WhisperX output.

    Uses each aligned segment's OWN text (no index remapping into
    faster-whisper's different segmentation) and includes the words array.

    Per-segment handling (NOT whole-request fallback):
    - Segments with valid aligned words: include words (after adaptation +
      exact text reconstruction check). stable-ts regroups/splits these.
    - Segments where alignment failed (no usable words): include a single
      synthetic word spanning the segment bounds. stable-ts DROPS segments
      that lack a non-empty words array, so we must provide one. Using a
      single word avoids inventing fake intra-segment timing boundaries.
    - Segments where words exist but text reconstruction FAILS: this is
      corruption (not just missing alignment) -> return ok=False to trigger
      whole-request faster-whisper fallback.

    Returns (result_segments, ok, synth_count). synth_count is the number
    of segments that used synthetic words (for observability/logging only;
    not encoded in the payload).

    This keeps WhisperX's segmentation as the primary path. Only failed-
    alignment segments use synthetic words, rather than discarding all
    aligned segments when a few fail.
    """
    result_segments = []
    synth_count = 0
    asg = aligned_segments.get("segments", [])
    for i, seg in enumerate(asg):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        text = seg.get("text", "")
        words = adapt_whisperx_words(seg.get("words") or [], seg_start, seg_end)
        if words and text.strip():
            # fail-safe: word concatenation must reproduce segment text
            words_text = "".join(w["word"] for w in words)
            if not _text_matches(words_text, text):
                return [], False, 0
        # When adaptation failed (words is None) but the segment has text,
        # generate a single synthetic word spanning the segment bounds.
        # stable-ts DROPS segments that lack a non-empty words array.
        # If synthesis fails (e.g. zero-duration segment after rounding),
        # the segment cannot be preserved -> treat as corruption -> whole-
        # request fallback.
        if words is None and text.strip():
            words = _synthesize_words_from_text(text, seg_start, seg_end)
            if not words:
                return [], False, 0
            synth_count += 1
        seg_out = {
            "id": i,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": text,
        }
        if words:
            seg_out["words"] = words
        result_segments.append(seg_out)
    return result_segments, True, synth_count


def build_result_from_faster_whisper(seg_list):
    """Fallback: use faster-whisper's own segments + word_timestamps words.

    Uses faster-whisper's OWN segmentation (not WhisperX's), so there is
    no index-mismatch between text and words.
    """
    result_segments = []
    for i, s in enumerate(seg_list):
        result_segments.append({
            "id": i,
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "text": s["text"],
            "words": faster_whisper_words_to_stable_ts(s.get("raw_words")),
        })
    return result_segments