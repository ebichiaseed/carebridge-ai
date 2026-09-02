'''
if you are using the AWS bedrock model (i.e Claude), inherit this model class
you can alter the temperature and max_tokens in the generate function to suit your needs
'''

import json
import boto3
from models.base_model import BaseModel


class BedrockModel(BaseModel):

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1"
    ):
        self.model_id = model_id

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        **kwargs
    ):

        response = self.client.converse(
            modelId=self.model_id,

            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ],

            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        )

        return response["output"]["message"]["content"][0]["text"]