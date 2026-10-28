# twitcasting-transcription

ツイキャス配信の録画ファイルから音声を解析し、SRT 字幕ファイルを生成する CLI ツールです。

## 機能

- 単一の動画ファイル、またはフォルダ内の動画を一括処理
- 各動画と同じフォルダに `.srt` を出力（1 動画 = 1 SRT）
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) による GPU 高速文字起こし
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) による話者識別（オプション）
- 既に SRT が存在するファイルのスキップ（`--skip-existing`）

## 対応形式

`.mp4` / `.mkv` / `.webm` / `.ts` / `.m4a`

## 必要環境

| 項目        | 要件                                  |
| ----------- | ------------------------------------- |
| Python      | 3.12 以上                             |
| ffmpeg      | PATH から実行可能であること           |
| GPU（推奨） | CUDA 対応 GPU（例: RTX 5070 Ti 16GB） |

### ffmpeg のインストール

```powershell
winget install ffmpeg
```

インストール後、ターミナルを開き直して `ffmpeg -version` が通ることを確認してください。

## セットアップ

[uv](https://docs.astral.sh/uv/) を使う場合:

```powershell
git clone <repository-url>
cd twitcasting_transcription
uv sync
```

話者識別を利用する場合は、追加の依存関係をインストールします。

```powershell
uv sync --extra diarization
```

さらに、[Community-1 の利用条件](https://huggingface.co/pyannote/speaker-diarization-community-1)に同意し、Hugging Face のアクセストークンを環境変数に設定してください。

```powershell
$env:HF_TOKEN = "hf_..."
```

## 使い方

### 単一ファイル

```powershell
uv run twitcasting-transcribe recording.mp4
```

`recording.srt` が同じフォルダに生成されます。

### フォルダ一括処理

```powershell
uv run twitcasting-transcribe C:\Twitcasting\recordings\
```

フォルダ内の動画ファイルを再帰的に検索し、それぞれに個別の SRT を出力します。1 本が失敗しても残りは処理を続け、失敗があると終了コード 1 で終わります。

```
recordings/
├── part1.mp4  →  part1.srt
├── part2.mp4  →  part2.srt
└── sub/
    └── extra.mp4  →  extra.srt
```

### 既存 SRT をスキップ

```powershell
uv run twitcasting-transcribe C:\recordings\ --skip-existing
```

## オプション

| オプション        | 短縮 | デフォルト | 説明                                            |
| ----------------- | ---- | ---------- | ----------------------------------------------- |
| `--model`         | `-m` | `large-v3` | Whisper モデル名                                |
| `--language`      | `-l` | `ja`       | 認識言語                                        |
| `--device`        | `-d` | `cuda`     | 推論デバイス（`cuda` / `cpu`）                  |
| `--compute-type`  |      | `float16`  | 量子化タイプ。CPU では `float16` / `int8_float16` を `int8` に切り替える |
| `--skip-existing` |      | off        | 既に `.srt` がある動画をスキップ                |
| `--diarize`       |      | off        | 話者を推定し字幕に話者ラベルを付ける            |
| `--diarization-model` |  | `pyannote/speaker-diarization-community-1` | 話者識別モデル |
| `--hf-token`      |      | `HF_TOKEN` | Hugging Face トークン。環境変数での指定を推奨   |
| `--min-speakers`  |      | 自動       | 想定する最小話者数                              |
| `--max-speakers`  |      | 自動       | 想定する最大話者数                              |

### 使用例

```powershell
# GPU + large-v3（デフォルト）
uv run twitcasting-transcribe recording.mp4

# CPU で実行（float16 は int8 に切り替わる）
uv run twitcasting-transcribe recording.mp4 --device cpu

# より軽いモデルで試す
uv run twitcasting-transcribe recording.mp4 --model medium

# 処理時間を短くする
uv run twitcasting-transcribe recording.mp4 --model large-v3-turbo

# 話者識別を有効にする
uv run twitcasting-transcribe recording.mp4 --diarize

# 話者数が2人と分かっている場合
uv run twitcasting-transcribe recording.mp4 --diarize --min-speakers 2 --max-speakers 2
```

`large-v3-turbo` はデコーダを縮めたモデルで、処理は短くなります。静かな発話では `large-v3` との差は小さく、雑音や固有名詞では `large-v3` の方が残ります。初回実行時にこのモデルもダウンロードされます。

話者識別した字幕には `[SPEAKER_00]` のようなラベルが付きます。これは話者の区別を表す推定ラベルであり、人物の実名を特定するものではありません。同時発話や短い相づちは誤判定されることがあります。

## 処理の流れ

1. 指定パスから動画ファイルを列挙
2. ffmpeg で 16kHz モノラル WAV に音声抽出
3. faster-whisper で文字起こし
4. `--diarize` 指定時は pyannote.audio で話者を推定
5. 動画と同じディレクトリに SRT を書き出し（発話がない場合は作らない。一時ファイルから置き換える）

初回実行時は Whisper モデル（デフォルト: `large-v3`）のダウンロードが行われます。

## パフォーマンスの目安

RTX 5070 Ti 16GB + `large-v3` + `float16` の場合（参考値）:

| 配信の長さ | 処理時間の目安 |
| ---------- | -------------- |
| 1 時間     | 約 5〜15 分    |
| 3 時間     | 約 15〜45 分   |

無音区間が多い配信では VAD フィルタにより、さらに短縮されることがあります。

## トラブルシューティング

### `ffmpeg not found`

ffmpeg がインストールされていないか、PATH に含まれていません。インストール後にターミナルを再起動してください。

### CUDA エラー（`cublas64_12.dll is not found`）

Windows では GPU 利用時に CUDA 12 の cuBLAS が必要です。`uv sync` 時に `nvidia-cublas-cu12` が自動インストールされ、アプリ起動時に DLL パスを登録します。

それでも解決しない場合:

- NVIDIA ドライバが最新か確認してください
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) をインストールし、再起動してください
- 一時的な回避策として `--device cpu` で実行できます（`int8` で動作し、処理は大幅に遅くなります）

### メモリ不足

`--model medium` や `--compute-type int8_float16` を試してください。

### 話者識別モデルの `401 Unauthorized`

[Community-1 の利用条件](https://huggingface.co/pyannote/speaker-diarization-community-1)に同意したHugging Faceアカウントでアクセストークンを作成し、実行するPowerShellで設定してください。

```powershell
$env:HF_TOKEN = "hf_..."
uv run twitcasting-transcribe recording.mp4 --diarize
```
