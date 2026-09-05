"""Cross-platform Transformers implementation of the Singlish Whisper model."""

import asyncio
from pathlib import Path
from typing import Any

from models.base_model import SpeechToTextModel
from models.transcription_agent.romanization import romanize_chinese_text


class TransformersWhisperSinglishModel(SpeechToTextModel):
    """Run the original Hugging Face checkpoint on CUDA or CPU."""

    def __init__(self, model_id: str, device: str):
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "Transformers is missing. Run: pip install -r requirements.txt"
            ) from error

        self.model_id = model_id
        self.device = device
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
        )

    async def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        file_path = Path(audio_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        result = await asyncio.to_thread(self._pipeline, str(file_path), **kwargs)
        return romanize_chinese_text(result["text"])
