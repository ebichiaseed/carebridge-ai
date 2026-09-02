# models/whisper_singlish.py

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

from models.base_model import SpeechToTextModel


class WhisperSinglishModel(SpeechToTextModel):

    def __init__(
        self,
        model_id="mjwong/whisper-large-v3-turbo-singlish"
    ):

        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype
        )

        processor = AutoProcessor.from_pretrained(
            model_id
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device
        )

    async def transcribe(self, audio_path: str, **kwargs):

        result = self.pipe(audio_path)

        return result["text"]