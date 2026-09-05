"""Apple Silicon MLX implementation of the multilingual Qwen ASR model."""

import asyncio
from pathlib import Path
from typing import Any

from models.base_model import SpeechToTextModel


DEFAULT_MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"


class QwenAsrModel(SpeechToTextModel):
    """Transcribe multilingual audio with Qwen3-ASR through MLX-Audio."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        try:
            from mlx_audio.stt import load
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MLX-Audio is missing. Run: pip install -r requirements.txt"
            ) from error
        except ImportError as error:
            raise RuntimeError(
                "MLX-Audio could not initialize. This backend requires an "
                "Apple Silicon Mac with Metal GPU access. Set ASR_BACKEND=cpu "
                "or cuda on another system."
            ) from error

        self.model_id = model_id
        self._model = load(model_id)

    async def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        """Transcribe audio, auto-detecting the spoken language by default."""
        file_path = Path(audio_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        result = await asyncio.to_thread(self._model.generate, str(file_path), **kwargs)
        return result.text
