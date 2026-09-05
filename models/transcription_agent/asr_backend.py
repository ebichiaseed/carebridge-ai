"""Choose a local ASR implementation that matches the host hardware."""

import platform
from enum import StrEnum


class AsrBackend(StrEnum):
    MLX = "mlx"
    CUDA = "cuda"
    CPU = "cpu"


def detect_asr_backend(override: str = "auto") -> AsrBackend:
    """Return the requested backend, or the best supported local backend."""
    if override != "auto":
        try:
            return AsrBackend(override)
        except ValueError as error:
            choices = ", ".join(["auto", *(backend.value for backend in AsrBackend)])
            raise ValueError(f"Unknown ASR backend {override!r}. Choose one of: {choices}.") from error

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return AsrBackend.MLX

    try:
        import torch
    except ImportError:
        return AsrBackend.CPU

    return AsrBackend.CUDA if torch.cuda.is_available() else AsrBackend.CPU
