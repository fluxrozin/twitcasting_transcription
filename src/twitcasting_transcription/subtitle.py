from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def format_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> int:
    lines: list[str] = []
    cue_index = 1
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.speaker:
            text = f"[{segment.speaker}] {text}"
        lines.append(str(cue_index))
        lines.append(
            f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}"
        )
        lines.append(text)
        lines.append("")
        cue_index += 1

    if not lines:
        return 0

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text("\n".join(lines), encoding="utf-8")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return cue_index - 1
