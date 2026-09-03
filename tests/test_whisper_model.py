import asyncio
from pathlib import Path
from unittest.mock import Mock
import unittest

from models.whisper_model import WhisperSinglishModel


class WhisperSinglishModelTests(unittest.TestCase):
    def test_transcribe_returns_pipeline_text_without_blocking(self):
        with self.subTest("existing audio file"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                audio_file = Path(directory) / "sample.wav"
                audio_file.touch()
                model = WhisperSinglishModel.__new__(WhisperSinglishModel)
                model.model_id = "test-model"
                model._transcribe = Mock(return_value={"text": "wah this one can"})

                text = asyncio.run(model.transcribe(str(audio_file)))

        self.assertEqual(text, "wah this one can")
        model._transcribe.assert_called_once_with(
            str(audio_file),
            path_or_hf_repo="test-model",
        )

    def test_transcribe_rejects_missing_audio_file(self):
        model = WhisperSinglishModel.__new__(WhisperSinglishModel)

        with self.assertRaisesRegex(FileNotFoundError, "Audio file not found"):
            asyncio.run(model.transcribe("does-not-exist.wav"))
