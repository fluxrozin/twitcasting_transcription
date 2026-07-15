from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".ts", ".m4a"}


def collect_videos(path: Path) -> list[Path]:
    if not path.exists():
        msg = f"Path does not exist: {path}"
        raise FileNotFoundError(msg)

    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            msg = f"Unsupported file type: {path.suffix}"
            raise ValueError(msg)
        return [path]

    videos = sorted(
        p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        msg = f"No video files found in: {path}"
        raise FileNotFoundError(msg)
    return videos


def output_srt_path(video: Path) -> Path:
    return video.with_suffix(".srt")
