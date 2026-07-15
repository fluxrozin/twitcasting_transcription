import subprocess
from pathlib import Path


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        msg = "ffmpeg not found. Install ffmpeg and ensure it is on PATH."
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        msg = f"ffmpeg failed for {video_path}: {stderr}"
        raise RuntimeError(msg) from exc
