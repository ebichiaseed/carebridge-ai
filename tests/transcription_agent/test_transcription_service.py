import asyncio
import unittest
from unittest.mock import call, patch

from services.transcription_agent.transcription_service import ParallelTranscriptionService


class FakeSpeechModel:
    def __init__(self, text: str):
        self.text = text

    async def transcribe(self, audio_path: str) -> str:
        await asyncio.sleep(0)
        return self.text


class ParallelTranscriptionServiceTests(unittest.TestCase):
    @patch("services.transcription_agent.transcription_service.romanize_chinese_text")
    def test_returns_raw_and_romanized_text_in_configured_order(self, romanize):
        romanize.side_effect = ["hello lah", "ni hao"]
        service = ParallelTranscriptionService(
            [
                ("whisper", FakeSpeechModel("hello lah")),
                ("qwen", FakeSpeechModel("你好")),
            ]
        )

        candidates = asyncio.run(service.transcribe("sample.wav"))

        self.assertEqual(
            [candidate.to_dict() for candidate in candidates],
            [
                {
                    "model": "whisper",
                    "text": "hello lah",
                    "raw_text": "hello lah",
                },
                {"model": "qwen", "text": "ni hao", "raw_text": "你好"},
            ],
        )
        romanize.assert_has_calls([call("hello lah"), call("你好")])
