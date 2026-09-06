# Shared application settings.

from dotenv import load_dotenv
import os

load_dotenv()


AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)

POLLY_VOICE_ID = os.getenv("POLLY_VOICE_ID", "Jasmine")
POLLY_ENGINE = os.getenv("POLLY_ENGINE", "neural")

INTERPRETATION_MODEL = os.getenv(
    "INTERPRETATION_MODEL",
    "sonnet"
)

TRANSLATION_MODEL = os.getenv(
    "TRANSLATION_MODEL",
    "sonnet"
)

VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "haiku")
