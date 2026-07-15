import tempfile
from pathlib import Path

import typer

from twitcasting_transcription.audio import extract_audio
from twitcasting_transcription.input_resolver import collect_videos, output_srt_path
from twitcasting_transcription.subtitle import write_srt
from twitcasting_transcription.transcriber import load_model, transcribe

def main(
    path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Video file or directory containing recordings.",
    ),
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model name."),
    language: str = typer.Option("ja", "--language", "-l", help="Transcription language."),
    device: str = typer.Option("cuda", "--device", "-d", help="Inference device."),
    compute_type: str = typer.Option(
        "float16",
        "--compute-type",
        help="Quantization type for faster-whisper.",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip videos that already have an SRT file.",
    ),
) -> None:
    videos = collect_videos(path)
    targets = [
        video
        for video in videos
        if not (skip_existing and output_srt_path(video).exists())
    ]

    if skip_existing:
        skipped = len(videos) - len(targets)
        if skipped:
            typer.echo(f"Skipping {skipped} file(s) with existing SRT.")

    if not targets:
        typer.echo("No files to process.")
        raise typer.Exit(code=0)

    typer.echo(f"Loading model: {model} ({device}, {compute_type})")
    whisper_model = load_model(
        model,
        device=device,
        compute_type=compute_type,
    )

    total = len(targets)
    for index, video in enumerate(targets, start=1):
        srt_path = output_srt_path(video)
        typer.echo(f"[{index}/{total}] {video.name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / f"{video.stem}.wav"
            typer.echo("  Extracting audio...")
            extract_audio(video, wav_path)

            typer.echo("  Transcribing...")
            segments = transcribe(whisper_model, wav_path, language=language)

        write_srt(segments, srt_path)
        typer.echo(f"  Wrote {srt_path}")

    typer.echo(f"Done. Processed {total} file(s).")


def app() -> None:
    typer.run(main)
