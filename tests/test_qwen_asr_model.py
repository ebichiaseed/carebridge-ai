import asyncio
from pathlib import Path
from unittest.mock import Mock
import unittest

from models.qwen_asr_model import QwenAsrModel


class QwenAsrModelTests(unittest.TestCase):
    def test_transcribe_returns_model_text(self):
        with self.subTest("existing audio file"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                audio_file = Path(directory) / "sample.wav"
                audio_file.touch()
                model = QwenAsrModel.__new__(QwenAsrModel)
                model._model = Mock()
                model._model.generate.return_value = Mock(text="我今天 okay lah")

                text = asyncio.run(model.transcribe(str(audio_file)))

        self.assertEqual(text, "我今天 okay lah")
        model._model.generate.assert_called_once_with(str(audio_file))

    def test_transcribe_rejects_missing_audio_file(self):
        model = QwenAsrModel.__new__(QwenAsrModel)

        with self.assertRaisesRegex(FileNotFoundError, "Audio file not found"):
            asyncio.run(model.transcribe("does-not-exist.wav"))
