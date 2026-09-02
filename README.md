# carebridge-ai


for now i shall use readme to introduce the repo to all, then later we amend nice nice

the repo skeleton:
```
backend/
│
├── agents/
│   ├── base_agent.py
│   ├── transcription_agent.py
│   ├── interpretation_agent.py
│   ├── translation_agent.py
│   └── ...
│
├── models/
│   ├── base_model.py
│   ├── bedrock_model.py
│   ├── whisper_model.py
│   └── model_factory.py
│
├── config/
│   └── settings.py
│
├── services/
│   └── ...
│
└── state.py
│
└── synthesiser.py

```


le important idea

```
Agent
  ↓
BaseAgent
  ↓
Model Interface
  ↓
┌─────────────────┬────────────────────┐
│ AWS Bedrock     │ Local/Hugging Face │
│ Claude/Nova/etc │ Whisper Singlish   │
└─────────────────┴────────────────────┘
```

so what do you need to do as agent specialists:

1. follow `agents_base_agent_example.py` for an idea on how to use the template

2. create your system prompt, create any tools where needed



and then how to access the AWS

1. ensure you have `awscli` downloaded, else download it

```
aws --version

brew install awscli
```

2. configure ur sso

refer to access keys for the info!

```
aws configure sso
```

3. configure profile name: hackathon

4. next time, when start working
```
aws sso login --profile hackathon
```