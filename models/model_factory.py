# models/model_factory.py

from configs.transcription_agent.settings import (
    ASR_BACKEND,
    QWEN_ASR_MLX_MODEL,
    QWEN_ASR_PORTABLE_MODEL,
    WHISPER_MLX_MODEL,
    WHISPER_PORTABLE_MODEL,
)
from configs.settings import AWS_REGION
from models.transcription_agent.asr_backend import AsrBackend, detect_asr_backend
from models.bedrock_model import BedrockModel
from models.transcription_agent.qwen_asr_model import QwenAsrModel
from models.transcription_agent.transformers_qwen_asr_model import TransformersQwenAsrModel
from models.transcription_agent.transformers_whisper_model import TransformersWhisperSinglishModel
from models.transcription_agent.whisper_model import WhisperSinglishModel


class ModelFactory:

    @staticmethod
    def create_whisper() -> WhisperSinglishModel | TransformersWhisperSinglishModel:
        backend = detect_asr_backend(ASR_BACKEND)
        if backend == AsrBackend.MLX:
            return WhisperSinglishModel(model_id=WHISPER_MLX_MODEL)
        return TransformersWhisperSinglishModel(
            model_id=WHISPER_PORTABLE_MODEL, device=backend.value
        )

    @staticmethod
    def create_qwen_asr() -> QwenAsrModel | TransformersQwenAsrModel:
        backend = detect_asr_backend(ASR_BACKEND)
        if backend == AsrBackend.MLX:
            return QwenAsrModel(model_id=QWEN_ASR_MLX_MODEL)
        return TransformersQwenAsrModel(
            model_id=QWEN_ASR_PORTABLE_MODEL, device=backend.value
        )

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
            return ModelFactory.create_whisper()
        elif model_name == "qwen-asr":
            return ModelFactory.create_qwen_asr()
        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )
