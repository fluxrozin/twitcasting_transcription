import tempfile
from pathlib import Path

import typer

from twitcasting_transcription.audio import extract_audio
from twitcasting_transcription.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    assign_speakers,
    diarize,
    load_diarization_pipeline,
)
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
    diarization: bool = typer.Option(
        False,
        "--diarize",
        help="Identify speakers and add speaker labels to subtitles.",
    ),
    diarization_model: str = typer.Option(
        DEFAULT_DIARIZATION_MODEL,
        "--diarization-model",
        help="pyannote speaker diarization model.",
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        envvar="HF_TOKEN",
        help="Hugging Face token (prefer the HF_TOKEN environment variable).",
    ),
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        min=1,
        help="Minimum expected number of speakers.",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        min=1,
        help="Maximum expected number of speakers.",
    ),
) -> None:
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise typer.BadParameter(
            "--min-speakers cannot be greater than --max-speakers."
        )

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
    diarization_pipeline = None
    if diarization:
        typer.echo(f"Loading diarization model: {diarization_model} ({device})")
        try:
            diarization_pipeline = load_diarization_pipeline(
                model_name=diarization_model,
                token=hf_token,
                device=device,
            )
        except RuntimeError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

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

            if diarization_pipeline is not None:
                typer.echo("  Identifying speakers...")
                turns = diarize(
                    diarization_pipeline,
                    wav_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                segments = assign_speakers(segments, turns)

        write_srt(segments, srt_path)
        typer.echo(f"  Wrote {srt_path}")

    typer.echo(f"Done. Processed {total} file(s).")


def app() -> None:
    typer.run(main)
