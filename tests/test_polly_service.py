import io
import unittest
from unittest.mock import Mock, patch

from services.polly_service import PollyService


class TestPollyService(unittest.TestCase):
    @patch("services.polly_service.boto3.client")
    def test_synthesizes_mp3_and_closes_stream(self, make_client):
        stream = io.BytesIO(b"mp3 audio")
        client = Mock()
        client.synthesize_speech.return_value = {"AudioStream": stream}
        make_client.return_value = client

        service = PollyService("ap-southeast-1", "Jasmine", "neural")
        result = service.synthesize("Please take your medicine.")

        self.assertEqual(result, b"mp3 audio")
        self.assertTrue(stream.closed)
        make_client.assert_called_once_with("polly", region_name="ap-southeast-1")
        client.synthesize_speech.assert_called_once_with(
            Text="Please take your medicine.",
            OutputFormat="mp3",
            VoiceId="Jasmine",
            Engine="neural",
        )


if __name__ == "__main__":
    unittest.main()
