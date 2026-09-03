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

## ASR demo: parallel Singlish and multilingual transcription

This portion of the project runs two local speech-to-text models on an Apple
Silicon Mac. Both receive the same audio file and return independent candidates
for a later word-level consensus stage.

```text
Audio file
  ├─ Singlish Whisper MLX      → English / Singlish candidate
  └─ Qwen3-ASR MLX 8-bit       → Mandarin / Cantonese / Minnan candidate
                                      ↓
                           JSON candidate output
```

### Requirements

- Apple Silicon Mac (M-series)
- 16 GB unified memory minimum; 24 GB or more is more comfortable
- Python 3.13 (the version used for the verified practice run)
- Homebrew, to install `ffmpeg`

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

`ffmpeg` reads common input formats such as WAV, MP3, and M4A. The direct Python
dependencies are pinned to the versions used for the verified practice run.

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
WHISPER_MODEL=wysie/whisper-large-v3-turbo-singlish-mlx
QWEN_ASR_MODEL=mlx-community/Qwen3-ASR-0.6B-8bit
AWS_REGION=ap-southeast-1
```

### Tests

```bash
python3 -m unittest discover -s tests -v
```

These unit tests validate file handling and concurrent candidate collection;
they do not prove real model accuracy or performance on your audio.
