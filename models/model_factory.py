# models/model_factory.py

from configs.settings import AWS_REGION, QWEN_ASR_MODEL, WHISPER_MODEL
from models.bedrock_model import BedrockModel
from models.qwen_asr_model import QwenAsrModel
from models.whisper_model import WhisperSinglishModel


class ModelFactory:

    @staticmethod
    def create(model_name: str):

        if model_name == "nova-lite":

            return BedrockModel(
                model_id="amazon.nova-lite-v1:0",
                region=AWS_REGION
            )

        elif model_name == "sonnet":

            return BedrockModel(
                model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
                region=AWS_REGION
            )

        elif model_name == "haiku":

            return BedrockModel(
                model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                region=AWS_REGION
            )
        elif model_name in {"whisper-singlish", WHISPER_MODEL}:
            return WhisperSinglishModel(model_id=WHISPER_MODEL)
        elif model_name in {"qwen-asr", QWEN_ASR_MODEL}:
            return QwenAsrModel(model_id=QWEN_ASR_MODEL)
        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )
