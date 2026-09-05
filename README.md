# carebridge-ai

for now i shall use readme to introduce the repo to all, then later we amend nice nice

the repo skeleton:
```
backend/
│
├── agents/
│   ├── base_agent.py
│   ├── transcription_agent.py
│   ├── interpretation_agent.py
│   ├── translation_agent.py
│   └── ...
│
├── models/
│   ├── base_model.py
│   ├── bedrock_model.py
│   ├── whisper_model.py
│   └── model_factory.py
│
├── config/
│   └── settings.py
│
├── services/
│   └── ...
│
└── state.py
│
└── synthesiser.py
```

le important idea
```
Agent
  ↓
BaseAgent
  ↓
Model Interface
  ↓
┌─────────────────┬────────────────────┐
│ AWS Bedrock     │ Local/Hugging Face │
│ Claude/Nova/etc │ Whisper Singlish   │
└─────────────────┴────────────────────┘
```

so what do you need to do as agent specialists:

1. follow `agents_base_agent_example.py` for an idea on how to use the template

2. create your system prompt, create any tools where needed



and then how to access the AWS

1. ensure you have `awscli` downloaded, else download it
```
aws --version

brew install awscli
```

2. configure ur sso

refer to access keys for the info!
```
aws configure sso
```

3. configure profile name: hackathon

4. next time, when start working
```
aws sso login --profile hackathon
```

### Transcription workflow layout

Files specific to `transcription_agent` belong in the matching layer directory
below. Top-level files remain shared infrastructure.

```text
agents/transcription_agent/   # transcription agent implementations
configs/transcription_agent/  # ASR backend and model settings
models/transcription_agent/   # Whisper, Qwen, and text-normalisation adapters
services/transcription_agent/ # parallel transcription service
tests/transcription_agent/    # transcription tests
```

## ASR demo: parallel Singlish and multilingual transcription

This portion of the project runs two local speech-to-text models. Both receive
the same audio file and return independent candidates
for a later word-level consensus stage. Chinese characters in either model's
output are converted to tone-less Mandarin pinyin; English, Singlish, and
punctuation are retained as returned by the model.

```text
Audio file
  ├─ Singlish Whisper          → English / Singlish candidate
  └─ Qwen3-ASR                 → Mandarin / Cantonese / Minnan candidate
                                      ↓
                           JSON candidate output
```

### Requirements

- Python 3.13
- `ffmpeg` available on your system
- 16 GB memory minimum is recommended; CUDA makes the portable models much
  faster, while CPU-only inference is supported but slower.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

`ffmpeg` reads common input formats such as WAV, MP3, and M4A. The installer
skips MLX dependencies except on Apple Silicon. Install the PyTorch build that
matches your NVIDIA CUDA version before installing the requirements if you want
GPU acceleration on Linux or Windows.

### Platform model selection

`ASR_BACKEND=auto` (the default) selects the models below:

| System | Whisper | Qwen ASR |
| --- | --- | --- |
| Apple Silicon | MLX conversion of the Singlish checkpoint | MLX 8-bit Qwen3-ASR |
| NVIDIA CUDA (Linux/Windows) | Original Singlish checkpoint via Transformers | Qwen3-ASR Transformers checkpoint |
| CPU-only Linux/Windows/Intel Mac | Original Singlish checkpoint via Transformers | Qwen3-ASR Transformers checkpoint |

The CUDA and CPU backends download their model weights on first use. Set
`ASR_BACKEND` to `mlx`, `cuda`, or `cpu` to override auto-detection, for example
when diagnosing a GPU installation.

### Run the included practice test

```bash
export HF_HUB_DISABLE_XET=1
python3 transcribe_parallel.py TestData/SinglishTestData.m4a
```

The first run downloads and caches both models:

- [`wysie/whisper-large-v3-turbo-singlish-mlx`](https://huggingface.co/wysie/whisper-large-v3-turbo-singlish-mlx) for Singlish and English.
- [`mlx-community/Qwen3-ASR-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) for multilingual transcription, including Chinese, Cantonese, and Minnan coverage.

The command prints two candidates as JSON:

```json
[
  {"model": "singlish_whisper", "text": "..."},
  {"model": "qwen_multilingual", "text": "..."}
]
```

`TestData/SinglishTestData.m4a` is version-controlled for this practice run;
other files under `TestData/` are ignored by default. A successful run produces
two non-empty candidate texts. Transcript wording may vary with model-runtime
updates, so use successful completion and output structure as the check rather
than a fixed transcript.

### Configuration

Copy `env_template` to `.env` only when you need to override model identifiers
or AWS settings. Do not commit `.env`.

```dotenv
ASR_BACKEND=auto
WHISPER_MLX_MODEL=wysie/whisper-large-v3-turbo-singlish-mlx
QWEN_ASR_MLX_MODEL=mlx-community/Qwen3-ASR-0.6B-8bit
WHISPER_PORTABLE_MODEL=mjwong/whisper-large-v3-turbo-singlish
QWEN_ASR_PORTABLE_MODEL=Qwen/Qwen3-ASR-0.6B-hf
AWS_REGION=ap-southeast-1
```

### Tests

```bash
python3 -m unittest discover -s tests -v
```

These unit tests validate backend selection, file handling, and concurrent
candidate collection; they do not prove real model accuracy or performance on
your audio.
