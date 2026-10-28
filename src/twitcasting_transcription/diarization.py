import logging
import wave
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from twitcasting_transcription.subtitle import SubtitleSegment

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def load_diarization_pipeline(
    *,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    token: str | None = None,
    device: str = "cuda",
) -> Any:
    try:
        torch_logger = logging.getLogger("torch.utils.flop_counter")
        previous_log_level = torch_logger.level
        torch_logger.setLevel(logging.ERROR)
        import torch
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*torchcodec is not installed correctly",
                category=UserWarning,
            )
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization requires the optional dependencies. "
            "Run `uv sync --extra diarization`."
        ) from exc
    finally:
        if "torch_logger" in locals():
            torch_logger.setLevel(previous_log_level)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "The installed PyTorch build does not support CUDA. "
            "Run `uv sync --extra diarization` to install the CUDA build."
        )

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except Exception as exc:
        message = f"Could not load the diarization model {model_name}: {exc}"
        if model_name == DEFAULT_DIARIZATION_MODEL:
            message += (
                " Accept its conditions at "
                "https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "and set the HF_TOKEN environment variable."
            )
        raise RuntimeError(message) from exc
    if pipeline is None:
        raise RuntimeError(
            f"Could not load the diarization model {model_name}. "
            "Accept its conditions on Hugging Face and set the HF_TOKEN environment variable."
        )

    if device == "cuda":
        pipeline.to(torch.device("cuda"))
    return pipeline


def _load_pcm_wav(audio_path: Path) -> dict[str, Any]:
    """Load the generated 16-bit PCM WAV without relying on TorchCodec."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization requires the optional dependencies. "
            "Run `uv sync --extra diarization`."
        ) from exc

    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise RuntimeError("Diarization expects a 16-bit PCM WAV file.")

    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    waveform = waveform.reshape(-1, channels).transpose(0, 1)
    waveform = waveform.to(dtype=torch.float32) / 32768.0
    return {"waveform": waveform, "sample_rate": sample_rate}


def _speaker_annotation(output: Any) -> Any:
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", None)
    if annotation is None:
        return output
    return annotation


def diarize(
    pipeline: Any,
    audio_path: Path,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    kwargs: dict[str, int] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"TensorFloat-32 \(TF32\) has been disabled",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"std\(\): degrees of freedom is <= 0",
            category=UserWarning,
        )
        output = pipeline(_load_pcm_wav(audio_path), **kwargs)
    annotation = _speaker_annotation(output)

    return [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, speaker in annotation
    ]


def assign_speakers(
    segments: Iterable[SubtitleSegment],
    turns: Iterable[SpeakerTurn],
) -> list[SubtitleSegment]:
    turns = list(turns)
    result: list[SubtitleSegment] = []
    for segment in segments:
        overlaps: dict[str, float] = {}
        for turn in turns:
            overlap = max(
                0.0,
                min(segment.end, turn.end) - max(segment.start, turn.start),
            )
            if overlap:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap

        speaker = max(overlaps, key=overlaps.get) if overlaps else None
        result.append(replace(segment, speaker=speaker))
    return result
