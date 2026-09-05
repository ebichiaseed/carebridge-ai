import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from models.transcription_agent.asr_backend import AsrBackend, detect_asr_backend
from models.model_factory import ModelFactory
from models import model_factory


class AsrBackendTests(unittest.TestCase):
    def test_uses_mlx_on_apple_silicon(self):
        with (
            patch("models.transcription_agent.asr_backend.platform.system", return_value="Darwin"),
            patch("models.transcription_agent.asr_backend.platform.machine", return_value="arm64"),
        ):
            self.assertEqual(detect_asr_backend(), AsrBackend.MLX)

    def test_uses_cuda_when_available_off_apple_silicon(self):
        torch = ModuleType("torch")
        torch.cuda = Mock(is_available=Mock(return_value=True))
        with (
            patch("models.transcription_agent.asr_backend.platform.system", return_value="Linux"),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            self.assertEqual(detect_asr_backend(), AsrBackend.CUDA)

    def test_uses_cpu_without_cuda(self):
        torch = ModuleType("torch")
        torch.cuda = Mock(is_available=Mock(return_value=False))
        with (
            patch("models.transcription_agent.asr_backend.platform.system", return_value="Windows"),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            self.assertEqual(detect_asr_backend(), AsrBackend.CPU)

    def test_factory_selects_the_matching_implementation(self):
        with (
            patch("models.model_factory.detect_asr_backend", return_value=AsrBackend.CUDA),
            patch("models.model_factory.TransformersWhisperSinglishModel") as whisper,
            patch("models.model_factory.TransformersQwenAsrModel") as qwen,
        ):
            ModelFactory.create("whisper-singlish")
            ModelFactory.create("qwen-asr")

        whisper.assert_called_once_with(
            model_id=model_factory.WHISPER_PORTABLE_MODEL, device="cuda"
        )
        qwen.assert_called_once_with(
            model_id=model_factory.QWEN_ASR_PORTABLE_MODEL, device="cuda"
        )
