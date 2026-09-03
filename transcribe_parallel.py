"""Print independent Singlish Whisper and Qwen ASR transcript candidates.

Usage:
    python3 transcribe_parallel.py /path/to/audio.wav
"""

import argparse
import asyncio
import json

from configs.settings import QWEN_ASR_MODEL, WHISPER_MODEL
from models.qwen_asr_model import QwenAsrModel
from models.whisper_model import WhisperSinglishModel
from services.transcription_service import ParallelTranscriptionService


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a file concurrently with Singlish Whisper and Qwen ASR."
    )
    parser.add_argument("audio_path", help="Path to a local audio file.")
    args = parser.parse_args()

    service = ParallelTranscriptionService(
        [
            ("singlish_whisper", WhisperSinglishModel(WHISPER_MODEL)),
            ("qwen_multilingual", QwenAsrModel(QWEN_ASR_MODEL)),
        ]
    )
    candidates = await service.transcribe(args.audio_path)
    print(json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
