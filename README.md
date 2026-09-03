# CareBridge AI

CareBridge AI currently provides a local, two-model speech-to-text demo for Singaporean multilingual audio. Both models run on an Apple Silicon Mac through MLX and produce independent transcript candidates for the same audio file.

## Current transcription flow

```text
Audio file
  ├─ Singlish Whisper MLX      → English / Singlish candidate
  └─ Qwen3-ASR MLX 8-bit       → Mandarin / Cantonese / Minnan candidate
                                      ↓
                           JSON candidate output
```

The candidates are deliberately returned separately. A word-level consensus stage, potentially using AWS Bedrock only for ambiguous spans, has not been implemented yet.

## Requirements

- Apple Silicon Mac (M-series)
- 16 GB unified memory minimum; 24 GB or more is more comfortable
- Python 3.13 (the version used for the verified practice run)
- Homebrew, to install `ffmpeg`

## Setup

Create an isolated environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

`ffmpeg` is required to read common audio formats such as WAV, MP3, and M4A.
The direct Python dependencies are pinned to the versions used for the verified
practice run.

## Run the parallel transcription demo

Use the included 9.7-second practice clip:

```bash
python3 transcribe_parallel.py TestData/SinglishTestData.m4a
```

On the first run, the models are downloaded and cached locally:

- [`wysie/whisper-large-v3-turbo-singlish-mlx`](https://huggingface.co/wysie/whisper-large-v3-turbo-singlish-mlx) for Singlish and English.
- [`mlx-community/Qwen3-ASR-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) for multilingual transcription, including Chinese, Cantonese, and Minnan coverage.

The command prints one candidate from each model:

```json
[
  {
    "model": "singlish_whisper",
    "text": "..."
  },
  {
    "model": "qwen_multilingual",
    "text": "..."
  }
]
```

Do not treat either transcript as ground truth. Evaluate both against real representative recordings, particularly for Singapore Hokkien and code-switched speech.

## Reproduce the practice run on another Mac

`TestData/SinglishTestData.m4a` is version-controlled so another author can run
the same practice command after cloning. Other files in `TestData/` are ignored
by default to prevent accidental commits of recordings.

```bash
export HF_HUB_DISABLE_XET=1
python3 transcribe_parallel.py TestData/SinglishTestData.m4a
```

`HF_HUB_DISABLE_XET=1` uses Hugging Face's standard download path, which is a
useful fallback when Xet transfers are interrupted. A successful run prints a
JSON list containing exactly two non-empty candidate texts named
`singlish_whisper` and `qwen_multilingual`. Transcript wording can vary with
the audio and model-runtime updates, so treat the structure and successful
completion—not a fixed transcript—as the reproducibility check.

## Configuration

Copy `env_template` to `.env` if you need to override model identifiers or AWS settings. Do not commit `.env`.

```dotenv
WHISPER_MODEL=wysie/whisper-large-v3-turbo-singlish-mlx
QWEN_ASR_MODEL=mlx-community/Qwen3-ASR-0.6B-8bit
AWS_REGION=ap-southeast-1
```

## Project layout

```text
agents/                       Shared agent base classes
configs/settings.py           Environment-backed configuration
models/                       Bedrock and local speech-model adapters
services/transcription_service.py
                              Concurrent candidate collection
transcribe_parallel.py        Demo command-line entry point
tests/                        Unit tests without model downloads
```

## Tests

Run the fast unit tests without downloading models:

```bash
python3 -m unittest discover -s tests -v
```

These tests validate file handling and concurrent candidate collection. They do not prove real model accuracy, model-download success, or performance on your audio.

## AWS Bedrock

Bedrock remains available for the project’s text-model tasks, such as future ambiguity adjudication. The two ASR models above run locally and are not Bedrock-hosted models.

To use the existing Bedrock wrapper, configure AWS SSO once:

```bash
brew install awscli
aws configure sso
aws sso login --profile hackathon
```
