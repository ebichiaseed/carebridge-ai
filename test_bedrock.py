import boto3

session = boto3.Session(
    profile_name="hackathon",
    region_name="us-east-1",
)

bedrock = session.client("bedrock-runtime")

response = bedrock.converse(
    modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Say hello in one sentence."
                }
            ],
        }
    ],
)

print(
    response["output"]["message"]["content"][0]["text"]
)