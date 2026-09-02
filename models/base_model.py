# common model interface for all models

from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text based on the given prompt."""
        pass