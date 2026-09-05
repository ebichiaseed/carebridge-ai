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

## Structure Agent

### Structure Agent evaluation

Six hand-written evaluation cases are defined in `tests/structure_test_cases.json`. Each case checks that the Structure Agent converts an `InterpretationResult` into natural language without changing safety-critical information such as timing, negation, object, or urgency.

The deterministic unit tests use a fake model and can be run with:

```bash
# Baseline strips out context
python3 -m tests.test_structure_agent --no-context \
  --results tests/structure_baseline_eval.jsonl

# Full evaluation passes context
python3 -m tests.test_structure_agent \
  --results tests/structure_full_eval.jsonl
```

Model-backed evaluation results can be recorded in:

- `tests/structure_baseline_eval.jsonl` — evaluation without recent context
- `tests/structure_full_eval.jsonl` — evaluation with recent context

The Structure Agent uses `temperature=0` and `max_tokens=300`.
#### What each case tests

| Case | Probe | What a failure would mean |
| --- | --- | --- |
| s001 | Preserve medication timing | Changing “after dinner” to “before dinner” could cause medication to be given at the wrong time. |
| s002 | Preserve medication negation | Losing or weakening the negation could turn a prohibited action into an instruction. |
| s003 | Preserve high urgency | Omitting “immediately” or “now” could delay emergency assistance. |
| s004 | Do not invent an action for a statement | Turning an observation into advice or an instruction would add information that was not supplied. |
| s005 | Use recent context safely | Context may improve the wording, but it must not replace the resolved actor or requested object. |
| s006 | Preserve scoped negation and the alternative action | Applying the negation to the wrong pill could reverse which medication must be withheld and which should be given. |
| s007 | Preserve a named actor | Dropping or replacing “Siti” could assign the instruction to the wrong person. |
| s008 | Preserve low urgency | Escalating a routine statement into an emergency would misrepresent its urgency. |
| s009 | Preserve a question | Turning a question into a command would change the speaker’s intent. |
| s010 | Preserve a future appointment time | Changing the day or time could cause the appointment to be missed. |
| s011 | Preserve a completed action | Rewriting an action that already happened as a future instruction could lead to it being repeated. |
| s012 | Preserve a missed dose | Losing the negation or time scope could incorrectly imply that the medication was taken or that the issue is ongoing. |
| s013 | Preserve a compound request | Dropping either action would make the resulting instruction incomplete. |
| s014 | Preserve a fall as an urgent event | Rewriting an actual fall as a possible future fall would change both the event and its urgency. |
| s015 | Do not invent a diagnosis | Adding an unsupported medical cause would turn a reported symptom into an unverified diagnosis. |
| s016 | Preserve breathing difficulty and urgency | Weakening either the symptom or its urgency could delay necessary assistance. |
| s017 | Preserve a prohibition and its timing | Losing “must not” or changing “after midnight” could reverse an important restriction. |
| s018 | Use context without changing the resolved object | Context may supply a location, but it must not replace the requested blood-pressure monitor with another item. |
| s019 | Ignore conflicting older context | Previous details must not override the current medication or timing resolved by the Interpretation Agent. |
| s020 | Keep the output concise | Adding explanations, unrelated medical details, or excessive sentences would make the translation less direct. |

#### Unit-test coverage

`tests/test_structure_agent.py` contains nine unit tests covering:

- Parsing a valid JSON model response
- Parsing JSON wrapped in Markdown code fences
- Supporting dictionary-shaped model responses
- Rejecting responses that are not valid JSON
- Rejecting responses with a missing `draft_translation`
- Rejecting a `draft_translation` that is not a string
- Adding interpretation fields and recent context to the prompt
- Serializing missing recent context as an empty list
- Calling the model with deterministic settings: `temperature=0` and `max_tokens=300`

The unit tests use a fake model, so they do not call Bedrock or consume model tokens.

The same file also contains the model-backed evaluation runner. It loads the JSON test cases, runs each case with or without recent context, applies phrase-based assertions, records latency and failures, and writes one result record per case.

#### Results

| Metric | Baseline (no recent context) | Full pipeline |
| --- | ---: | ---: |
| Cases passing all automated assertions | 18/20 | **18/20** |
| Valid structured responses | 20/20 | **20/20** |
| Mean latency | 2.01 s | **1.90 s** |
| Median latency | 1.84 s | **1.87 s** |

Both runs generated a valid `draft_translation` for all 20 cases. Eighteen cases passed every automated assertion in each configuration.

Cases `s002` and `s006` were marked as failures because the evaluator accepts only “do not” or “don’t,” while the generated responses used semantically valid alternatives such as “should not” and “requests not to.” In `s006`, the forbidden-phrase check also matched the substring “give the blue pill tonight” inside the correctly negated sentence “should not give the blue pill tonight.”

The outputs therefore appear to preserve the intended negation, but the current phrase-based evaluator does not recognise all valid wording. These results should be reported as **18/20 automated assertion passes**, rather than 20/20, until the assertions are corrected and the evaluation is rerun.

The full pipeline was approximately 0.10 seconds faster in mean latency, although its median latency was approximately 0.02 seconds slower. This small difference is not sufficient to conclude that recent context consistently improves or reduces latency.

Each JSONL result record contains the case ID, probe, status, latency, assertion failures, and generated `draft_translation`, allowing baseline and full-context behaviour to be compared case by case.

#### Known limitations

The unit tests validate response parsing, prompt construction, deterministic model settings, and error handling, but they do not directly assess translation quality because they use canned fake-model responses.

The model-backed evaluation uses simple substring assertions. This can produce false failures when the model uses a valid synonym, such as “should not” instead of “do not,” or when a forbidden phrase occurs inside a correctly negated sentence. The current checker also does not enforce the `maximum_sentences` rule included in case `s020`.

The evaluation contains only 20 cases and one recorded generation per configuration. Because model output and latency can vary between runs, the results should not be treated as statistically conclusive. The negation assertions should be improved, sentence-count validation should be implemented, and the evaluation should then be repeated across multiple runs.