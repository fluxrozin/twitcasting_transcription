from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


def format_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}"
        )
        lines.append(segment.text.strip())
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
