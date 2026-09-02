# models/model_factory.py

from configs.settings import AWS_REGION
from models.bedrock_model import BedrockModel
from models.whisper_model import WhisperSinglishModel


class ModelFactory:

    @staticmethod
    def create(model_name: str):

        if model_name == "sonnet":

            return BedrockModel(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                region=AWS_REGION
            )

        elif model_name == "haiku":

            return BedrockModel(
                model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                region=AWS_REGION
            )
        elif model_name == "whisper-singlish":
                    return WhisperSinglishModel()
        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )