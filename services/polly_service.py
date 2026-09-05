"""Amazon Polly text-to-speech integration."""

from typing import BinaryIO

import boto3


class PollyService:
    """Synthesize short CareBridge messages as MP3 audio."""

    def __init__(self, region: str, voice_id: str, engine: str):
        self.voice_id = voice_id
        self.engine = engine
        self.client = boto3.client("polly", region_name=region)

    def synthesize(self, text: str) -> bytes:
        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId=self.voice_id,
            Engine=self.engine,
        )
        audio_stream: BinaryIO = response["AudioStream"]
        try:
            return audio_stream.read()
        finally:
            audio_stream.close()
