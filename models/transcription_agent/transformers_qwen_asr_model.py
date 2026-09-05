"""Cross-platform Transformers implementation of Qwen3-ASR."""

import asyncio
from pathlib import Path
from typing import Any

from models.base_model import SpeechToTextModel


class TransformersQwenAsrModel(SpeechToTextModel):
    """Run Qwen3-ASR through Transformers on CUDA or CPU."""

    def __init__(self, model_id: str, device: str):
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "PyTorch and Transformers are required. Run: pip install -r requirements.txt"
            ) from error

        self.model_id = model_id
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype)
        self._model.to(device).eval()

    async def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        file_path = Path(audio_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        return await asyncio.to_thread(self._transcribe_sync, str(file_path), **kwargs)

    def _transcribe_sync(self, audio_path: str, **kwargs: Any) -> str:
        inputs = self._processor.apply_transcription_request(audio=audio_path).to(
            self._model.device, self._model.dtype
        )
        output_ids = self._model.generate(**inputs, **kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        text = self._processor.decode(
            generated_ids, return_format="transcription_only"
        )[0]
        return text
