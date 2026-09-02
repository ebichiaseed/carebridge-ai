# config/settings.py

import os


AWS_REGION = os.getenv(
    "AWS_REGION",
    "ap-southeast-1"
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
    "mjwong/whisper-large-v3-turbo-singlish"
)