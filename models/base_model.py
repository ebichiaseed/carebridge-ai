# common model interface for all models

from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text based on the given prompt."""
        pass


class SpeechToTextModel(ABC):
    """Common interface for models that convert an audio file to text."""

    @abstractmethod
    async def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        """Transcribe an audio file and return its text."""
        pass
