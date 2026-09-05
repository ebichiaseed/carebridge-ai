# carebridge-ai

for now i shall use readme to introduce the repo to all, then later we amend nice nice

## Notes

### Repo Skeleton
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

### Agent
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

### Orchestration Workflow

overarching idea:

```
START
  ↓
interpretation_agent
  ├─ needs clarification ─────────────→ CLARIFY → END
  ↓
structure_agent
  ↓
verify_agent
  ├─ PASS ────────────────────────────→ END
  ├─ CLARIFY ─────────────────────────→ CLARIFY → END
  └─ RETRY and retry_count < limit ───→ structure_agent
```


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

## ASR demo: parallel Singlish and multilingual transcription

This portion of the project runs two local speech-to-text models on an Apple
Silicon Mac. Both receive the same audio file and return independent candidates
for a later word-level consensus stage.

```text
Audio file
  ├─ Singlish Whisper MLX      → English / Singlish candidate
  └─ Qwen3-ASR MLX 8-bit       → Mandarin / Cantonese / Minnan candidate
                                      ↓
                           JSON candidate output
```

### Requirements

- Apple Silicon Mac (M-series)
- 16 GB unified memory minimum; 24 GB or more is more comfortable
- Python 3.13 (the version used for the verified practice run)
- Homebrew, to install `ffmpeg`

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

`ffmpeg` reads common input formats such as WAV, MP3, and M4A. The direct Python
dependencies are pinned to the versions used for the verified practice run.

### Run the included practice test

```bash
export HF_HUB_DISABLE_XET=1
python3 transcribe_parallel.py TestData/SinglishTestData.m4a
```

The first run downloads and caches both models:

- [`wysie/whisper-large-v3-turbo-singlish-mlx`](https://huggingface.co/wysie/whisper-large-v3-turbo-singlish-mlx) for Singlish and English.
- [`mlx-community/Qwen3-ASR-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) for multilingual transcription, including Chinese, Cantonese, and Minnan coverage.

The command prints two candidates as JSON:

```json
[
  {"model": "singlish_whisper", "text": "..."},
  {"model": "qwen_multilingual", "text": "..."}
]
```

`TestData/SinglishTestData.m4a` is version-controlled for this practice run;
other files under `TestData/` are ignored by default. A successful run produces
two non-empty candidate texts. Transcript wording may vary with model-runtime
updates, so use successful completion and output structure as the check rather
than a fixed transcript.

### Configuration

Copy `env_template` to `.env` only when you need to override model identifiers
or AWS settings. Do not commit `.env`.

```dotenv
WHISPER_MODEL=wysie/whisper-large-v3-turbo-singlish-mlx
QWEN_ASR_MODEL=mlx-community/Qwen3-ASR-0.6B-8bit
AWS_REGION=ap-southeast-1
```

### Tests

```bash
python3 -m unittest discover -s tests -v
```

These unit tests validate file handling and concurrent candidate collection;
they do not prove real model accuracy or performance on your audio.

## Intepretation Agent and Glossary Lookup

**Unit tests** prove the glossary retriever's logic is correct. They inject a
deterministic fake embedder, so they need no AWS credentials and no network.

**The interpretation eval** measures how well the agent reads real Singlish and
Chinese caregiving utterances. It calls Bedrock, so it costs tokens and needs an
active SSO session.

### Unit tests

```bash
source .venv/bin/activate
python3 -m unittest tests.test_glossary_lookup -v    # 55 tests
python3 -m unittest discover -s tests -v             # 64 tests, full suite
```

| Suite | Tests | Result | Runtime |
| --- | --- | --- | --- |
| `tests.test_glossary_lookup` | 55 | all passing | 0.02 s |
| Full suite (`discover -s tests`) | 64 | all passing | 0.04 s |

The glossary suite covers nine areas:

| Area | What it guards |
| --- | --- |
| Alias matching | spacing and punctuation variants, spelling variants, word boundaries, longest-form ranking |
| Script-aware length floor | 2-character CJK forms (头晕, 跌倒, 中风) stay matchable while 2-character Latin forms stay excluded |
| Vector search | paraphrase retrieval, threshold cutoff, alias-over-vector ranking, no duplicate entries |
| Input validation | bad parameters rejected, `None` candidates from a failed ASR model skipped rather than fatal |
| Degradation | index failure and query-time failure both leave alias matching working; `lookup()` never raises |
| Contract | hit shape matches what the agent reads; `to_trace()` counts alias and vector hits separately |
| File loading | missing file lists every path searched; malformed JSON reports line and column |
| Validation | duplicate ids, unknown fields, missing fields, empty glossary all rejected |
| Embedding cache | reuse on second index, corrupt cache falls back, wrong-dimension vectors dropped, no temp file left behind |

The remaining 9 tests cover the verify agent, both ASR models and the parallel
transcription service.

### Glossary retrieval configuration

45 curated entries across seven categories (`negation`, `symptom`, `tcm`,
`time`, `referent`, `daily_living`, `urgency_state`). Retrieval is hybrid:
exact alias matching first, then Titan v2 vector search for paraphrases.

```bash
python3 -m tools.glossary_lookup --check       # offline: load, validate, alias smoke tests
python3 -m tools.glossary_lookup --calibrate   # needs SSO: sweeps the similarity threshold
```

`MIN_SIMILARITY = 0.28`, calibrated against 14 probes. Vector-only true
positives scored 0.301–0.413; the highest false positive scored 0.267. The
band is narrow, so re-run `--calibrate` after any material glossary edit —
entry text changes the embedding and shifts the scores.

### Interpretation agent evaluation

20 hand-written cases in `tests/interp_test_cases.json`. Each probes one
specific way the interpretation node can fail, and several are deliberately
paired so that the contrast between two near-identical inputs isolates which
input is doing the work.

```bash
python3 -m tests.test_interp_eval --cases tests/interp_test_cases.json --no-glossary
python3 -m tests.test_interp_eval --cases tests/interp_test_cases.json
```

Model: Claude Sonnet via Bedrock, `temperature=0`.

#### What each case tests

| Case | Probe | What a failure would mean |
| --- | --- | --- |
| c001 | Singlish "never" as past-tense negation | Reading it as "not ever" turns a missed dose into a claim she never takes medicine at all |
| c002 | "ah girl" with no person info | Guessing "daughter" or "helper" means the model invents referents it cannot know |
| c003 | Same utterance, person info supplied | Pairs with c002 — shows person info resolving what the glossary cannot |
| c004 | 受不了 with no symptom named | Escalating to high urgency means pain is being inferred from the phrase alone |
| c005 | 受不了 with the symptom named | Pairs with c004 — urgency should rise and the ambiguity note should clear |
| c006 | 酸 as dull ache, not sharp pain | Reading soreness as sharp pain over-escalates a routine complaint |
| c007 | 热气 (TCM heatiness) must not become fever | Inventing a clinical finding from a folk concept |
| c008 | "send" means accompany, not dispatch | Dropping the caregiver from the instruction changes who must act |
| c009 | "later you fall down" as a warning | Treating a caution as a scheduled future event |
| c010 | 等下 — vague timing on a medication dose | Unresolved dose timing that is not flagged is a silent medication risk |
| c011 | Two ASR candidates disagreeing across scripts | Flagship case: glossary, person info and recent context must all contribute to one reconciled reading |
| c012 | An exclamation with no action | Raising `InterpretationError` on a normal remark breaks the graph |
| c013 | A paiseh refusal that context contradicts | Flattening a face-saving refusal into a plain "no" hides a real need |
| c014 | A fall that has already happened | Urgency must be high; nothing downstream can recover it if this is missed |
| c015 | "jialat" with no symptom named | Escalating on an expressive particle alone |
| c016 | 辛苦 — tired-from-work or unwell | Committing to one sense of an ambiguous term without saying so |
| c017 | Negation scoped to one specific object | `negated: true` with no scope loses half of "not the blue one, the white one" |
| c018 | One-sided weakness (stroke sign) | Whether a directive glossary note lifts urgency on a red-flag symptom |
| c019 | Two appliance actions in one utterance | Whether a compound request survives the schema or is silently truncated |
| c020 | A pronoun resolvable only from the previous turn | Shows recent context doing work neither glossary nor person info can |

#### Results

| | Baseline (no glossary) | Full pipeline |
| --- | --- | --- |
| Cases passing all asserts | 15/20 | **18/20** |
| Schema validation pass rate | 20/20 | 20/20 |
| Mean latency | 3.6 s | 4.2 s |
| Median latency | 2.9 s | 4.1 s |

Both runs use the same prompt and the same model, so the only variable is
retrieval. Three cases were fixed and none regressed, at a cost of roughly
1.2 s of median latency per turn:

| Case | Probe | Baseline | Full |
| --- | --- | --- | --- |
| c002 | unresolvable "ah girl" | referent not flagged | fixed |
| c010 | 等下 — vague timing on a dose | ambiguity not flagged | fixed |
| c013 | paiseh refusal masking a real need | ambiguity not flagged | fixed |

c005 also improved without passing: the baseline additionally misread a named
knee-pain complaint as `statement` rather than `distress`, which the glossary
corrected.

No safety-critical failures in either run. Negation is correct on every case,
urgency is `high` on both the fall (c014) and the one-sided weakness (c018), no
clinical findings are invented, and no referent is fabricated.

#### Known limitations

Both remaining failures are ambiguity-flagging precision, not meaning errors.
Both fail in the baseline too, so neither is caused by retrieval.

| Case | Behaviour | Assessment |
| --- | --- | --- |
| c005 | Correct urgency, correct `distress`, but adds a note that 她 is unidentified | Knowing *which* woman has knee pain does not change what the listener must do. The note is redundant rather than wrong. |
| c020 | 那个 is correctly resolved to 尿布 from recent context, then flagged as "not named explicitly" | A resolution reported as an ambiguity. Harmless but inflates the clarification rate. |

Prompt revisions aimed at either case regress c008 and c019, which are requests
with an unnamed addressee and fire the same rule. Recorded here rather than
chased further within the timebox.

#### Trace output

Each run writes one JSON record per case to `tests/interp_runs.jsonl`,
including the glossary trace (`tool_success`, alias vs vector match counts,
latency) and the interpret trace (`schema_valid`, `utterance_type`, `negated`,
`urgency`, `ambiguity_count`). This is the evidence behind the schema-validity,
tool-success and clarification-rate metrics in the observability plan.