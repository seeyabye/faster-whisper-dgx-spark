"""Parakeet TDT backend for whisper-fw.

Loads NVIDIA Parakeet-tdt-0.6b-v3 via transformers AutoModelForTDT and
converts its TDT (Token-and-Duration Transducer) output to the standard
segments[*].{id, start, end, text, words[*].{word, start, end, probability}}
schema that stable-ts and the /asr endpoint expect.

Key differences from FasterWhisperBackend:
- Native word timestamps via TDT (no WhisperX/wav2vec2 alignment needed)
- Sub-word token output merged into words
- ~4x less GPU memory, ~7x faster inference
- English + 24 other languages
"""

import logging

logger = logging.getLogger(__name__)


# ---- Pure conversion functions (testable without torch/transformers) ---- #


def tokens_to_words(tokens):
    """Merge sub-word TDT tokens into word-level dicts.

    TDT tokens have leading spaces for non-first words (e.g.
    'W', 'ild', 'ings', ' do', ' a'). A new word starts when
    a token begins with a space. Zero-duration punctuation tokens
    are appended to the current word, not emitted standalone.
    """
    words = []
    current_word = ""
    current_start = None
    current_end = None

    for tok in tokens:
        text = tok.get("token", "")
        if not text:
            continue
        ts = float(tok.get("start", 0))
        te = float(tok.get("end", ts))

        if current_start is None:
            current_start = ts
            current_end = te
            current_word = text
        elif text.startswith(" "):
            words.append({
                "word": current_word,
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "probability": 1.0,
            })
            current_word = text
            current_start = ts
            current_end = te
        else:
            current_word += text
            current_end = te

    if current_word:
        words.append({
            "word": current_word,
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "probability": 1.0,
        })

    return words


def words_to_segments(words, full_text="", pause_threshold=0.5):
    """Split words into segments by sentence-ending punctuation OR pauses.

    Creates segments at '.', '?', '!' boundaries, OR when the gap between
    consecutive words exceeds pause_threshold (default 0.5s). This handles
    cases where the model drops punctuation between speakers or during
    silence gaps (e.g. "But how Thoros?" -> "But how?" + "Thoros?").

    Each segment's first word is normalized to remove any leading space so
    that ''.join(w['word']) reproduces segment.text exactly.
    """
    segments = []
    current_words = []
    current_start = None
    prev_end = None
    seg_id = 0

    def _flush(words_list, start_val, sid):
        """Normalize first word (strip leading space) and build segment."""
        norm = [dict(words_list[0])]
        if norm[0]["word"].startswith(" "):
            norm[0]["word"] = norm[0]["word"].lstrip()
        norm.extend(words_list[1:])
        seg_text = "".join(ww["word"] for ww in norm)
        return {
            "id": sid,
            "start": round(start_val, 3),
            "end": round(words_list[-1]["end"], 3),
            "text": seg_text,
            "words": norm,
        }

    for w in words:
        # Check for pause-based split (gap between previous word end
        # and current word start exceeds threshold)
        if prev_end is not None and current_words:
            gap = w["start"] - prev_end
            if gap >= pause_threshold:
                segments.append(_flush(current_words, current_start, seg_id))
                seg_id += 1
                current_words = []
                current_start = None

        if current_start is None:
            current_start = w["start"]
        current_words.append(w)
        prev_end = w["end"]

        wt = w["word"].rstrip()
        if wt.endswith(".") or wt.endswith("?") or wt.endswith("!"):
            segments.append(_flush(current_words, current_start, seg_id))
            seg_id += 1
            current_words = []
            current_start = None
            prev_end = None

    if current_words:
        segments.append(_flush(current_words, current_start, seg_id))

    return segments


# ---- Backend class (requires torch + transformers) ---- #


class ParakeetBackend:
    """ASR backend using NVIDIA Parakeet TDT via transformers."""

    def __init__(self, model_id="nvidia/parakeet-tdt-0.6b-v3", device="cuda"):
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        self.model_id = model_id
        self.device = device
        self._torch = torch
        logger.info(f"Loading Parakeet model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Use float16 instead of bfloat16 — bfloat16 causes cuDNN LSTM
        # errors on some GPUs (CUDNN_STATUS_NOT_SUPPORTED).
        self.model = AutoModelForTDT.from_pretrained(
            model_id, dtype=torch.float16
        ).to(device)
        self.model.eval()
        logger.info(
            f"Parakeet loaded. GPU mem: "
            f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )

    def transcribe(self, audio_path, language=None, task="transcribe",
                   word_timestamps=True, **kwargs):
        """Transcribe audio and return (segments, full_text).

        Returns segments in the standard schema:
            [{id, start, end, text, words: [{word, start, end, probability}]}]
        """
        from faster_whisper.audio import decode_audio

        # Decode to 16kHz mono float32 (handles arbitrary uploads)
        audio = decode_audio(audio_path, sampling_rate=16000)

        inputs = self.processor(audio, sampling_rate=16000)
        inputs.to(self.model.device, dtype=self.model.dtype)

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs, return_dict_in_generate=True
            )

        decoded_output, decoded_timestamps = self.processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )

        full_text = (
            decoded_output[0]
            if isinstance(decoded_output, list)
            else decoded_output
        )
        if isinstance(full_text, list):
            full_text = full_text[0]

        # Convert TDT tokens to words
        tokens = decoded_timestamps[0] if isinstance(
            decoded_timestamps, list
        ) else decoded_timestamps
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]

        words = tokens_to_words(tokens)

        # Verify text reconstruction
        concat = "".join(w["word"] for w in words)
        if concat != full_text:
            logger.warning(
                f"Parakeet text reconstruction mismatch: "
                f"concat={concat[:60]!r} vs text={full_text[:60]!r}"
            )

        # Split into sentence-level segments
        segments = words_to_segments(words, full_text)

        return segments, full_text