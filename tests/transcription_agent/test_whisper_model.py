import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from models.transcription_agent.whisper_model import WhisperSinglishModel


class WhisperSinglishModelTests(unittest.TestCase):
    @patch("models.transcription_agent.whisper_model.romanize_chinese_text")
    def test_transcribe_romanizes_pipeline_text_without_blocking(
        self, romanize_chinese_text
    ):
        romanize_chinese_text.return_value = "wah ni hao"
        with self.subTest("existing audio file"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                audio_file = Path(directory) / "sample.wav"
                audio_file.touch()
                model = WhisperSinglishModel.__new__(WhisperSinglishModel)
                model.model_id = "test-model"
                model._transcribe = Mock(return_value={"text": "wah 你好"})

                text = asyncio.run(model.transcribe(str(audio_file)))

        self.assertEqual(text, "wah ni hao")
        model._transcribe.assert_called_once_with(
            str(audio_file),
            path_or_hf_repo="test-model",
        )
        romanize_chinese_text.assert_called_once_with("wah 你好")

    def test_transcribe_rejects_missing_audio_file(self):
        model = WhisperSinglishModel.__new__(WhisperSinglishModel)

        with self.assertRaisesRegex(FileNotFoundError, "Audio file not found"):
            asyncio.run(model.transcribe("does-not-exist.wav"))
