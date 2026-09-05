"""Run independent speech-to-text models concurrently."""

import asyncio
from dataclasses import asdict, dataclass
from typing import Sequence

from models.base_model import SpeechToTextModel


@dataclass(frozen=True)
class TranscriptCandidate:
    model: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ParallelTranscriptionService:
    """Collect independent hypotheses before a later consensus step."""

    def __init__(self, models: Sequence[tuple[str, SpeechToTextModel]]):
        self.models = tuple(models)

    async def transcribe(self, audio_path: str) -> list[TranscriptCandidate]:
        texts = await asyncio.gather(
            *(model.transcribe(audio_path) for _, model in self.models)
        )
        return [
            TranscriptCandidate(model=name, text=text)
            for (name, _), text in zip(self.models, texts, strict=True)
        ]
