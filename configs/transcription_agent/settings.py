"""Configuration for local transcription backends and model checkpoints."""

import os

from dotenv import load_dotenv


load_dotenv()


ASR_BACKEND = os.getenv("ASR_BACKEND", "auto")

WHISPER_MLX_MODEL = os.getenv(
    "WHISPER_MLX_MODEL",
    "wysie/whisper-large-v3-turbo-singlish-mlx",
)

QWEN_ASR_MLX_MODEL = os.getenv(
    "QWEN_ASR_MLX_MODEL",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
)

WHISPER_PORTABLE_MODEL = os.getenv(
    "WHISPER_PORTABLE_MODEL",
    "mjwong/whisper-large-v3-turbo-singlish",
)

QWEN_ASR_PORTABLE_MODEL = os.getenv(
    "QWEN_ASR_PORTABLE_MODEL",
    "Qwen/Qwen3-ASR-0.6B-hf",
)
