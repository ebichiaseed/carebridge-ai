import asyncio
import unittest

from services.transcription_agent.transcription_service import ParallelTranscriptionService


class FakeSpeechModel:
    def __init__(self, text: str):
        self.text = text

    async def transcribe(self, audio_path: str) -> str:
        await asyncio.sleep(0)
        return self.text


class ParallelTranscriptionServiceTests(unittest.TestCase):
    def test_returns_one_candidate_per_model_in_configured_order(self):
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
                {"model": "whisper", "text": "hello lah"},
                {"model": "qwen", "text": "你好"},
            ],
        )
