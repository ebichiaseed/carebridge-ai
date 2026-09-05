# Shared application settings.

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

VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "sonnet")
