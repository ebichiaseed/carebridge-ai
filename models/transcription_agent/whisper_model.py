"""Apple Silicon MLX implementation of the Singlish Whisper ASR model."""

import asyncio
from pathlib import Path
from typing import Any

from models.base_model import SpeechToTextModel
from models.transcription_agent.romanization import romanize_chinese_text


DEFAULT_MODEL_ID = "wysie/whisper-large-v3-turbo-singlish-mlx"


class WhisperSinglishModel(SpeechToTextModel):
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        """Configure the model; MLX downloads and caches weights on first use."""
        try:
            import mlx_whisper
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MLX Whisper is missing. Run: pip install -r requirements.txt"
            ) from error
        except ImportError as error:
            raise RuntimeError(
                "MLX Whisper could not initialize. This backend requires an "
                "Apple Silicon Mac with Metal GPU access. Set ASR_BACKEND=cpu "
                "or cuda on another system."
            ) from error

        self.model_id = model_id
        self._transcribe = mlx_whisper.transcribe

    async def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        """Transcribe an existing local audio file.

        Extra keyword arguments are forwarded to ``mlx_whisper.transcribe``,
        for example ``word_timestamps=True`` when word timestamps are needed.
        """
        file_path = Path(audio_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        result = await asyncio.to_thread(
            self._transcribe,
            str(file_path),
            path_or_hf_repo=self.model_id,
            **kwargs,
        )

        return romanize_chinese_text(result["text"])
