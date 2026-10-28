import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path

_WINDOWS_CUDA_DLL_PATHS_KEY = "TWITCASTING_CUDA_DLL_PATHS"


def _ensure_windows_cuda_dll_paths() -> None:
    if sys.platform != "win32" or os.environ.get(_WINDOWS_CUDA_DLL_PATHS_KEY):
        return

    dll_dirs: list[Path] = []
    for package in ("nvidia.cublas", "nvidia.cuda_nvrtc"):
        try:
            bin_dir = files(package) / "bin"
        except (ModuleNotFoundError, TypeError, FileNotFoundError):
            continue
        if not bin_dir.is_dir():
            continue
        dll_dirs.append(bin_dir)

    if dll_dirs:
        path_prefix = os.pathsep.join(str(dll_dir) for dll_dir in dll_dirs)
        os.environ["PATH"] = path_prefix + os.pathsep + os.environ.get("PATH", "")
        for dll_dir in dll_dirs:
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass

    try:
        ctranslate2_dir = files("ctranslate2")
        for dll_dir in dll_dirs:
            for dll_path in dll_dir.glob("*.dll"):
                target = ctranslate2_dir / dll_path.name
                if not target.is_file():
                    shutil.copy2(dll_path, target)
    except (ModuleNotFoundError, TypeError, FileNotFoundError, OSError):
        pass

    os.environ[_WINDOWS_CUDA_DLL_PATHS_KEY] = "1"


_ensure_windows_cuda_dll_paths()

from faster_whisper import WhisperModel

from twitcasting_transcription.subtitle import SubtitleSegment

# Same limits faster-whisper uses before it resamples a window.
_MAX_COMPRESSION_RATIO = 2.4
_MIN_AVG_LOGPROB = -1.0
_MIN_REPEAT_CHARS = 12


def _stable_segments(segments) -> list[SubtitleSegment]:
    """Drop repetitive, low-confidence, and long repeated cues."""
    kept: list[SubtitleSegment] = []
    previous: str | None = None
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if (
            segment.compression_ratio > _MAX_COMPRESSION_RATIO
            or segment.avg_logprob < _MIN_AVG_LOGPROB
        ):
            continue
        if previous == text and len(text) >= _MIN_REPEAT_CHARS:
            continue
        previous = text
        kept.append(SubtitleSegment(start=segment.start, end=segment.end, text=text))
    return kept


def load_model(
    model_name: str,
    *,
    device: str = "cuda",
    compute_type: str = "float16",
) -> WhisperModel:
    if device == "cuda" and sys.platform == "win32":
        _ensure_windows_cuda_dll_paths()
        cublas_dll = files("ctranslate2") / "cublas64_12.dll"
        if not cublas_dll.is_file():
            raise RuntimeError(
                "CUDA runtime library cublas64_12.dll is missing. "
                "Run `uv sync` to install nvidia-cublas-cu12, or use --device cpu."
            )
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    *,
    language: str = "ja",
) -> list[SubtitleSegment]:
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    return _stable_segments(segments)
