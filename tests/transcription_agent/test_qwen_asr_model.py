import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from models.transcription_agent.qwen_asr_model import QwenAsrModel


class QwenAsrModelTests(unittest.TestCase):
    @patch("models.transcription_agent.qwen_asr_model.romanize_chinese_text")
    def test_transcribe_romanizes_model_text(self, romanize_chinese_text):
        romanize_chinese_text.return_value = "wo jin tian okay lah"
        with self.subTest("existing audio file"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                audio_file = Path(directory) / "sample.wav"
                audio_file.touch()
                model = QwenAsrModel.__new__(QwenAsrModel)
                model._model = Mock()
                model._model.generate.return_value = Mock(text="我今天 okay lah")

                text = asyncio.run(model.transcribe(str(audio_file)))

        self.assertEqual(text, "wo jin tian okay lah")
        model._model.generate.assert_called_once_with(str(audio_file))
        romanize_chinese_text.assert_called_once_with("我今天 okay lah")

    def test_transcribe_rejects_missing_audio_file(self):
        model = QwenAsrModel.__new__(QwenAsrModel)

        with self.assertRaisesRegex(FileNotFoundError, "Audio file not found"):
            asyncio.run(model.transcribe("does-not-exist.wav"))
