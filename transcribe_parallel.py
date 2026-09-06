"""Print independent Singlish Whisper and Qwen ASR transcript candidates.

Usage:
    python3 transcribe_parallel.py /path/to/audio.wav
"""

import argparse
import asyncio
import json

from models.model_factory import ModelFactory
from services.transcription_agent.transcription_service import ParallelTranscriptionService


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a file concurrently with Singlish Whisper and Qwen ASR."
    )
    parser.add_argument("audio_path", help="Path to a local audio file.")
    args = parser.parse_args()

    service = ParallelTranscriptionService(
        [
            ("singlish_whisper", ModelFactory.create("whisper-singlish")),
            ("qwen_multilingual", ModelFactory.create("qwen-asr")),
        ]
    )
    candidates = await service.transcribe(args.audio_path)
    print(json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
