# config/settings.py

from dotenv import load_dotenv
import os

load_dotenv()


AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)

INTERPRETATION_MODEL = os.getenv(
    "INTERPRETATION_MODEL",
    "sonnet"
)

TRANSLATION_MODEL = os.getenv(
    "TRANSLATION_MODEL",
    "sonnet"
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "wysie/whisper-large-v3-turbo-singlish-mlx"
)

QWEN_ASR_MODEL = os.getenv(
    "QWEN_ASR_MODEL",
    "mlx-community/Qwen3-ASR-0.6B-8bit"
)

VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "haiku")
