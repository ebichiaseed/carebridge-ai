# CareBridge

## Problem Statement

A caregiver in his/her first few months caring for a Hokkien- or Cantonese-speaking elderly employer needs a way to understand daily-care instructions, be it medication, meals, mobility, in the moment they're given, because misunderstanding them causes confusion and distress on both sides. This gap is real enough that NTUC and the Centre for Domestic Employees launched dedicated Hokkien and Cantonese classes for migrant domestic workers in December 2024 specifically to address it (NTUC/CDE, Dec 2024), building on earlier findings that caregivers may experience embarrassment and humiliation when unable to understand their employers (McKay, 2013). This problem also extends to caregivers in general, where a miscommunication may potentially be life-threatening such as when symptoms are down-played due to the translation gap.

## Motivation
Caregivers in Singapore are mostly trained in English and often cannot speak the dialect of the elderly Chinese employers whom they care for. This is an acute gap especially in the first few months of their placement as they have yet to informally picked up on dialect phrases - and for foreign domestic workers (FDWs) Singlish slang - cultivated through constant communication. FDW interviewees have described the embarrassment and humiliation this causes when they struggle to understand instructions from their employers (McKay, 2013), risking confusion or frustration when needs and demands are misread in daily caregiving routines.
This gap is already recognised at a national level: in December 2024, NTUC and the Centre for Domestic Employees began running Hokkien and Cantonese classes for domestic workers. One participant, Enik Suparmi, who has worked for over 25 years in Singapore, described the direct caregiving impact: "It helps me a lot, especially to communicate with grandma... when my grandma orders things from me, asks me to cook... I can say yes, okay" (NTUC/CDE, Dec 2024). But these pilot classes are small (11 participants in the Hokkien pilot, 25 in the Cantonese pilot) and take months to build fluency. This leaves a real-time gap for FDWs who are mid-placement, not yet enrolled, or facing a dialect/idiom the curriculum doesn't cover. Local caregivers whose primary language may be English can potentially struggle with speaking Chinese or dialects, hence also facing the same issue as FDWs. This communication gap is what we're proposing to close, which we believe will benefit all the caregivers.

## Key Features
- **Parallel speech recognition:** Processes the same audio using separate models for English and Singlish, as well as Mandarin, Cantonese, and Hokkien.
- **Caregiving-aware interpretation:** Identifies requests, statements, symptoms, timing, negation, urgency, and referenced people or objects.
- **Glossary-assisted understanding:** Uses a curated caregiving glossary and semantic retrieval to interpret local expressions, dialect terms, and common ambiguities.
- **Context-aware translation:** Uses recent conversation context and known information about the people involved when interpreting an utterance.
- **Safety verification:** Checks whether important information was preserved before accepting a translation.
- **Targeted retries:** Re-runs only the affected stage when the verifier detects an interpretation or translation problem.
- **Clarification questions:** Requests clarification when an ambiguity could materially change the meaning or required action.
- **Speech playback:** Reads completed English translations aloud using Amazon Polly's Singapore English `Jasmine` voice, and Mandarin `Zhiyu` voice
- **Local observability:** Records each completed workflow as a local JSON file for evaluation and debugging.

## How It Works

```text
Spoken instruction
        ↓
Parallel transcription
        ↓
Caregiving-aware interpretation
        ↓
Structured English draft
        ↓
Safety verification
   ┌────┴──────────────────────────────┐
   ↓                                   ↓
Accept                        Retry the faulted stage
   ↓                                   ↓
Display Agent renders             Verify again
for the reader's language              ↓
   ↓                          Clarify if unresolved
Read aloud (Polly)
```

## Instructions to start up the app

Please ensure that you have these parameters in the .env file
```plain text
AWS_PROFILE=hackathon
AWS_REGION=us-east-1
POLLY_VOICE_ID=Jasmine
POLLY_ENGINE=neural
```
The AWS role behind that profile must allow `polly:SynthesizeSpeech`. CareBridge
uses Polly's Singapore English `Jasmine` and Mandarin `Zhiyu` neural voice for the **Read translation
aloud** button. AWS credentials remain on the server and are never sent to the
browser.

Assuming AWS is set up, copy and paste these commands into the terminal

```
aws sso login --profile hackathon
export AWS_PROFILE=hackathon
uvicorn frontend:app --reload
```

[The above example assumes that the profile name is ‘hackathon’, which requires sso configuration]

Each completed translation workflow is written locally as a separate JSON file
under `run_logs/`. Set `CAREBRIDGE_RUN_LOG_DIR` to use a different directory.
The directory is ignored by Git because logs may contain private conversation
content.

## Technical Overview

### Repo Skeleton

```
.
├── frontend.py                      FastAPI server and API routes
├── index.html                       Browser interface served by FastAPI
├── synthesiser.py                   Interpretation -> structure -> verification workflow
├── agents/
│   ├── interpretation_agent.py      Meaning, urgency, and ambiguity (Claude Sonnet)
│   ├── structure_agent.py           English draft translation (Claude Sonnet)
|   ├── display_agent.py             Post-graph rendering into the reader's language (Claude Haiku)
│   └── verify_agent.py              Meaning-preservation check and retry (Claude Haiku)
├── models/
│   ├── bedrock_model.py             Claude Sonnet, Claude Haiku, and Nova Lite access
│   └── transcription_agent/         Singlish Whisper and multilingual Qwen3-ASR adapters
├── services/                        Parallel transcription, Polly, and run logging
│   ├── polly_service.py             Amazon Polly TTS
│   ├── run_log_service.py           Per-run JSON logs under run_logs/
│   └── transcription_agent/         Parallel ASR
├── configs/                         Runtime and ASR model configuration
├── tools/                           Caregiving glossary and hybrid retrieval
├── tests/                           Unit tests, fixtures, and recorded evaluations
├── env_template.txt                 Environment-variable template
└── requirements.txt                 Python dependencies
```


### Orchestration Workflow

A concise mental model:
```
Interpret → Structure → Verify → Accept
    ↑           ↑          |
    |           |          ├─ interpretation fault → retry Interpret
    |           └──────────┴─ structure fault → retry Structure
    |
important ambiguity → best-effort Structure → translation + Clarify
unrecoverable/error/exhausted retry → Clarify

```

The full implementation:
```
START
  ↓
INTERPRET
  ├─ agent/parsing failure ───────────────→ CLARIFY → END
  ├─ important ambiguity detected
  │      ↓
  │   STRUCTURE BEST-EFFORT
  │      ↓
  │   return translation + question ─────→ END
  └─ sufficiently clear
         ↓
      STRUCTURE
         ├─ failure/empty draft ──────────→ CLARIFY → END
         ↓
      VERIFY
         ├─ PASS ─────────────────────────→ ACCEPT → END
         │
         ├─ RETRY: interpretation fault
         │    and retries remain
         │      ↓
         │    increment retry_count
         │      ↓
         │    INTERPRET → STRUCTURE → VERIFY
         │
         ├─ RETRY: structure/translation fault
         │    and retries remain
         │      ↓
         │    increment retry_count
         │      ↓
         │    STRUCTURE → VERIFY
         │
         ├─ CLARIFY ──────────────────────→ CLARIFY → END
         │
         └─ RETRY with budget exhausted ─→ CLARIFY → END

```

### Agent

```
Agent
  ↓
BaseAgent
  ↓
Model Interface
  ↓
┌──────────────────────┬──────────────────────────┐
│ AWS Bedrock          │ Local / Amazon Polly     │
│ Interpretation Agent │ Whisper Singlish (MLX)   │
│ Structure Agent      │ Qwen3-ASR Multilingual   │
│ Verify Agent         │   (MLX, Apple Silicon)   │
│ Display Agent        │                          │
│ (Claude Sonnet 4.5 / │                          │
│  Claude Haiku 4.5)   │                          │
└──────────────────────┴──────────────────────────┘
```


## Description of the Agents

## 1. Transcription Agent

### Transcription workflow layout

Files specific to `transcription_agent` belong in the matching layer directory
below. Top-level files remain shared infrastructure.

```text
agents/transcription_agent/   # transcription agent implementations
configs/transcription_agent/  # ASR backend and model settings
models/transcription_agent/   # Whisper, Qwen, and text-normalisation adapters
services/transcription_agent/ # parallel transcription service
tests/transcription_agent/    # transcription tests
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

`ffmpeg` reads common input formats such as WAV, MP3, and M4A. The installer
skips MLX dependencies except on Apple Silicon. Install the PyTorch build that
matches your NVIDIA CUDA version before installing the requirements if you want
GPU acceleration on Linux or Windows.

The `.env` file should be configured as below:

```dotenv
ASR_BACKEND=auto
WHISPER_MLX_MODEL=wysie/whisper-large-v3-turbo-singlish-mlx
QWEN_ASR_MLX_MODEL=mlx-community/Qwen3-ASR-0.6B-8bit
WHISPER_PORTABLE_MODEL=mjwong/whisper-large-v3-turbo-singlish
QWEN_ASR_PORTABLE_MODEL=Qwen/Qwen3-ASR-0.6B-hf
AWS_REGION=us-east-1
```

`ASR_BACKEND=auto` (the default) selects the models below:

| System | Whisper | Qwen ASR |
| --- | --- | --- |
| Apple Silicon | MLX conversion of the Singlish checkpoint | MLX 8-bit Qwen3-ASR |
| NVIDIA CUDA (Linux/Windows) | Original Singlish checkpoint via Transformers | Qwen3-ASR Transformers checkpoint |
| CPU-only Linux/Windows/Intel Mac | Original Singlish checkpoint via Transformers | Qwen3-ASR Transformers checkpoint |

The CUDA and CPU backends download their model weights on first use. Set
`ASR_BACKEND` to `mlx`, `cuda`, or `cpu` to override auto-detection, for example
when diagnosing a GPU installation.

### Test

```bash
export HF_HUB_DISABLE_XET=1
python3 transcribe_parallel.py TestData/SinglishTestData.m4a
```
The command prints two candidates as JSON:

```json
[
  {"model": "singlish_whisper", "text": "..."},
  {"model": "qwen_multilingual", "text": "..."}
]
```


## 2. Interpretation Agent

### Interpretation pipeline (glossary retrieval + interpretation agent)

The agent and glossary scripts can be ran without orchestration as it is an upstream node.

```
transcript(s) ──► GlossaryRetriever.lookup() ──► hits ──┐
                  alias match ∪ vector search           ├──► InterpretationAgent.run()
person_info ────────────────────────────────────────────┤        │
recent_context ─────────────────────────────────────────┘        ▼
                                              InterpretationResult (Pydantic)
```

### Files

| Path | Role |
| --- | --- |
| `tools/glossary_lookup.py` | Hybrid retriever, glossary schema/validation, embedding cache, CLI diagnostics |
| `tools/caregiving_glossary.json` | Curated glossary (more info below) |
| `agents/interpretation_agent.py` | `InterpretationResult` schema, system prompt, `needs_clarification()`, `to_trace()` |
| `tests/test_glossary_lookup.py` | Unit tests — no network, no AWS, injected `FakeEmbedder` |
| `tests/interp_test_cases.json` | Probe cases with `assert` / `watch_for` / `expect_ambiguity` |

### Glossary

45 entries, 248 matchable surface forms, 38 carrying an `ambiguity` note.
Categories: `daily_living` 13, `symptom` 11, `negation` 6, `urgency_state` 6,
`referent` 4, `tcm` 3, `time` 2.

Example of a glossary term:
```json
{
  "id": "g001",
  "term": "never",
  "language": "Singlish",
  "meaning": "marks that something DID NOT happen on this occasion; past-tense negation, not 'not ever'",
  "variants": ["never did", "never take"],
  "category": "negation",
  "example": "she never take her medicine this morning",
  "ambiguity": "Translating it as 'not ever' turns a missed dose into a refusal of all medication."
}
```

`extra="forbid"` on the entry model, so a typo'd key fails at startup rather than at
query time. `_validate_entries` reports every problem in one pass. `_content_warnings`
flags non-fatal issues (an entry with no alias-matchable form or a form with two entries).

### Retrieval

Two paths, unioned, alias always ranked first:

1. **Alias match** — deterministic regex over `term` + `variants`. Punctuation and
   spacing insensitive (`paiseh` / `pai seh` / `pai-seh` all match), word-boundary
   anchored for Latin forms so `song` does not fire inside `belong`. Scores 1.0.
   Works with no network.
2. **Cosine similarity** — Titan Text Embeddings v2 (`amazon.titan-embed-text-v2:0`,
   1024-dim, `normalize=True`, so dot product *is* cosine). Earns its keep on
   paraphrases and on a CJK ASR candidate matching an English-language entry.
   Entries already alias-matched are excluded, so no hit is duplicated.

Length floor is script-aware: **3 characters for Latin, 2 for CJK**. A flat floor of 3
silently drops 头晕, 跌倒, 不要, 中风, which is roughly half the glossary. `_has_latin` casefolds
first, otherwise uppercase-only forms like `GP`/`BP` take the CJK branch and lose their
word-boundary anchors.

Pass **all** ASR candidates to `lookup()`, not just one. Each candidate is embedded
separately — mean-pooling a Singlish and a Mandarin candidate produces a vector that
matches neither.

Config knobs at the top of the module: `TOP_K = 3`, `MIN_SIMILARITY = 0.28`,
`MAX_HITS = 8`.

### Failure behaviour

- A missing or malformed glossary raises `GlossaryConfigError` **at construction**. A
  silent fallback to a stub glossary mid-demo is worse than failing visibly at startup.
- `lookup()` never raises. If Bedrock is unreachable or credentials have expired, vector
  search is dropped and alias matching still answers; `result.success` goes False and
  `result.error` carries the exception name.
- Zero hits on a transcript containing no glossary terms is a **success**, not a tool
  failure — this matters for Tool-Call Success Rate.
- AWS errors are translated into actionable messages (expired SSO, model access not
  enabled, wrong region).

### Embedding cache

On-disk JSON beside the glossary, keyed by SHA-256 of `model_id | dimensions |
embed_text`. Changing the model, the dimensions or an entry's text invalidates that key
instead of mixing vector spaces. Corrupt or wrong-dimension cache entries are dropped
and re-embedded. Writes are atomic via a `.tmp` + `replace`.

### Interpretation agent

Please do not redefine the nodes. Every field added since the first draft has a default, so callers that ignore them still work.

```python
utterance_type: "request" | "statement" | "question" | "distress"
actor, action, object, timing: str | None      # action is optional: "aiyo, so hot today"
negated: bool
negation_cue: str | None                       # "mai", "bo", "don't"
negation_scope: str | None                     # "don't give the BLUE one, give the white"
urgency: "low" | "normal" | "high"
ambiguity: list[str]
clarification_question: str | None
source_language, cleaned_transcript: str | None
```

Some important prompt rules:

- Vector hits are labelled `(paraphrase match)` and framed as candidate readings. A
  discarded candidate must **not** produce an ambiguity note.
- Person info resolves **who**, never which **sense** of a term applies. Knowing 阿嬤 is
  Mdm Tan does not settle whether 辛苦 means tired or unwell.
- `urgency` defaults to `"normal"`. `"low"` is only for an explicit statement of
  non-concern — never for "nothing urgent was mentioned".
- A referent resolved from context is resolved. Re-flagging it as ambiguous is the
  failure mode the referent rules exist to prevent.
- Only `PROFILE_FIELDS` (`preferred_name`, `languages`, `household_terms`,
  `relationships`, `communication_preferences`) ever reach a prompt. No raw profile dict
  is f-stringed in. Clinically relevant vocabulary belongs in the glossary `ambiguity`
  field, not in the profile.
- `temperature=0` — this is structured extraction, not generation.

Invalid model output raises `InterpretationError`. The node routes to CLARIFY; it never
fabricates a result.

`needs_clarification(result)` is deliberately stricter than "ambiguity is non-empty" —
clarifying on every flagged ambiguity tanks Clarification Precision. It fires only on
high urgency, negation with an unresolved scope, or a request/distress turn missing its
action or object.

### Observability

Both halves emit trace dicts with no chain-of-thought and no transcript text:

- `GlossaryLookupResult.to_trace()` → `tool_success`, `matches`, `alias_matches`,
  `vector_matches`, `vector_search_used`, `latency_ms`, `error_code`
- `interpretation_agent.to_trace()` → `schema_valid`, `utterance_type`, `negated`,
  `urgency`, `ambiguity_count`, `needs_clarification`

### Usage

Instantiate the singleton once in the FastAPI startup hook. The first call embeds every
uncached entry and must not sit inside a user request.

```python
from tools.glossary_lookup import get_retriever

retriever = get_retriever()          # loads, validates, indexes

result = await retriever.alookup(transcript)   # alookup: lookup does blocking I/O
state["glossary_hits"] = result.hits
```

### Known gaps

- The two-character Hokkien negation `bo` sits below the Latin alias floor of 3 and is
  invisible to the retriever. Lowering the floor makes `mai`-class forms match across
  word breaks (`\s*` can span one), so this is unresolved rather than forgotten.
- `MIN_SIMILARITY = 0.28` is calibrated against the probe set in `_CALIBRATION_PROBES`,
  not against real ASR traces.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

These unit tests validate file handling and concurrent candidate collection, and not used to test for real-world accuracy.

**Unit tests** prove the glossary retriever's logic is correct. They inject a
deterministic fake embedder, so they need no AWS credentials and no network.

**The interpretation eval** measures how well the agent reads real Singlish and
Chinese caregiving utterances.

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

## 3. Structure Agent

### Structure Agent evaluation

Twenty hand-written evaluation cases are defined in `tests/structure_test_cases.json`. Each case checks that the Structure Agent converts an `InterpretationResult` into natural language without changing safety-critical information such as timing, negation, object, or urgency.

The deterministic unit tests use a fake model and need no AWS:

```bash
python3 -m unittest tests.test_structure_agent -v
```

The model-backed evaluation is a separate entry point in the same file and does
call Bedrock:

```bash
# Baseline strips out context
python3 -m tests.test_structure_agent --no-context \
  --results tests/structure_baseline_eval.jsonl

# Full evaluation passes context
python3 -m tests.test_structure_agent \
  --results tests/structure_full_eval.jsonl
```

The Structure Agent uses `temperature=0` and `max_tokens=300`.
#### What each case tests

| Case | Probe | What a failure would mean |
| --- | --- | --- |
| s001 | Preserve medication timing | Changing “after dinner” to “before dinner” could cause medication to be given at the wrong time. |
| s002 | Preserve medication negation | Losing or weakening the negation could turn a prohibited action into an instruction. |
| s003 | Preserve high urgency | Omitting “immediately” or “now” could delay emergency assistance. |
| s004 | Do not invent an action for a plain statement | Turning an observation into advice or an instruction would add information that was not supplied. |
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
| s016 | Preserve breathing difficulty | Weakening either the symptom or its urgency could delay necessary assistance. |
| s017 | Preserve a prohibition and its timing | Losing “must not” or changing “after midnight” could reverse an important restriction. |
| s018 | Use context to produce a resolved location | Context may supply a location, but it must not replace the requested blood-pressure monitor with another item. |
| s019 | Ignore conflicting older context | Previous details must not override the current medication or timing resolved by the Interpretation Agent. |
| s020 | Keep the output concise | Adding explanations, unrelated medical details, or excessive sentences would make the translation less direct. |

#### Unit-test coverage

`tests/test_structure_agent.py` contains eleven unit tests covering:

- Parsing a valid JSON model response
- Parsing JSON wrapped in Markdown code fences
- Supporting dictionary-shaped model responses
- Rejecting responses that are not valid JSON
- Rejecting responses with a missing `draft_translation`
- Rejecting a `draft_translation` that is not a string
- Adding interpretation fields and recent context to the prompt
- Serializing missing recent context as an empty list
- Calling the model with deterministic settings: `temperature=0` and `max_tokens=300`
- Passing verifier feedback into the retry prompt (`verification_issues`)
- Requiring English output in the prompt ("natural English sentences", "Do not return Chinese")

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

The results file is a JSON array with one record per case: case ID, probe, status,
latency, assertion failures, and the generated `draft_translation`

#### Known limitations

The unit tests validate response parsing, prompt construction, deterministic model settings, and error handling, but they do not directly assess translation quality because they use canned fake-model responses.

The model-backed evaluation uses simple substring assertions. This can produce false failures when the model uses a valid synonym, such as “should not” instead of “do not,” or when a forbidden phrase occurs inside a correctly negated sentence. The current checker also does not enforce the `maximum_sentences` rule included in case `s020`.

The evaluation contains only 20 cases and one recorded generation per configuration. Because model output and latency can vary between runs, the results should not be treated as statistically conclusive. The negation assertions should be improved, sentence-count validation should be implemented, and the evaluation should then be repeated across multiple runs.

## 4. Verify Agent

This is a safety gate. It sees the original transcript, the `InterpretationResult`, and the
draft translation, and decides whether critical meaning survived. It does not rewrite
anything, and only returns a verdict. On failure, it names which upstream agent must redo
its work.

### Files

| Path | Role |
| --- | --- |
| `agents/verify_agent.py` | `VerificationResult` schema, system prompt, JSON parsing with fail-safe |
| `tests/test_verify_agent.py` | Unit tests with a fake model (no Bedrock needed) |
| `tests/structure_test_cases.json` | Shared with the Structure Agent for meaning-preservation probes |

### Schema

```python
class VerificationResult(PydanticModel):
    verdict: Literal["PASS", "RETRY", "CLARIFY"]
    issues: list[str] = []
    fault_source: Literal[
        "interpretation", "structure", "translation", "source_ambiguity"
    ] | None = None
    clarification_question: str | None = None
```

`fault_source` is what makes retries targeted rather than a blind re-run of the whole
pipeline. `"translation"` is retained for backwards compatibility only; new responses
should use `"structure"`.

### Verdict rules

| Condition | verdict | fault_source | Router action |
| --- | --- | --- | --- |
| Meaning preserved | `PASS` | `null` | → ACCEPT |
| Interpretation wrong, transcript clear | `RETRY` | `interpretation` | re-run INTERPRET → STRUCTURE → VERIFY |
| Interpretation right, draft dropped/distorted meaning | `RETRY` | `structure` | re-run STRUCTURE → VERIFY |
| Transcript or interpretation itself ambiguous | `CLARIFY` | `source_ambiguity` | → CLARIFY |

Checked fields: actor, action, object, timing, negation, urgency. Every `RETRY` issue
must state what the named upstream agent should correct — a bare "translation is wrong"
gives the retry nothing to act on.

### Failure behaviour

`BedrockModel.generate()` returns plain text, not a forced tool-call shape, so the
response is parsed and validated here. `_parse_result` strips markdown fences, then on
`JSONDecodeError` or `ValidationError` returns `CLARIFY` with the parse error as an
issue and a generic clarification question.

The agent **never raises** and **never invents a PASS**. An unparseable verifier
response degrades to asking the user, not to silently accepting an unchecked draft.
This is the graph's "no infinite loop, no silent crash" invariant.

### Unit tests

`tests/test_verify_agent.py` uses a `FakeModel` returning canned strings:

- valid `PASS` JSON parses to the expected result
- `RETRY` carries its `fault_source` through
- unparseable text fails safe to `CLARIFY` with a non-empty `issues` list
- a response wrapped in ```` ```json ```` fences still parses

### Known limitations

The verifier reasons over an English draft against a possibly non-English transcript.
Cross-language meaning comparison is the harder half of its job; this is why the Display
Agent runs after the graph rather than inside it (see below).

Verifier catch rate is measured by injecting deliberately corrupted translations, not by
observing organic failures — the numbers in the orchestrator evaluation reflect that
setup.

---

## 5. Display Agent

Presentation layer, not a pipeline stage. It runs **after** the graph and feeds nothing
back into it.


## Orchestrator Evaluation Results

## Summary

The production Bedrock synthesiser performed strongly and passed every enforced quality threshold across 30 multilingual test cases.

| Metric | Result | Threshold |
|---|---:|---:|
| Task completion | 96.7% (29/30) | 75% |
| Interpretation accuracy | 86.7% (26/30) | 75% |
| Semantic equivalence | 96.7% (29/30) | 75% |
| Loop discipline | 100% | 100% |
| Schema validation | 100% | 100% |
| Verifier false-pass rate | 0% | — |

Key findings:

- The only overall failure was **T23**. For “If she feels unwell, call me,” the system asked who “she” referred to instead of completing the translation. It therefore returned `needs_clarification` when `verified` was expected.
- Four interpretation classifications failed:
  - **T13** and **T27:** classified statements as requests.
  - **T23** and **T26:** did not mark contextual negative expressions such as “unwell” or “has not eaten” as negation.
- All five retry/repair attempts succeeded: **T05, T16, T18, T26, and T30**.
- Safety-sensitive preservation was generally excellent:
  - Sequence: 100%
  - Quantity: 100%
  - Required negation: 94.1%
  - Unsupported information: 0%
- The independent judge found a few minor wording losses despite accepting the overall meaning:
  - **T14:** “coughing a lot” was weakened to “coughing.”
  - **T25:** “already” was omitted.
  - **T27:** “now” was introduced without being present in the expected result.

Performance and cost:

- Average latency: **9.5 seconds per case**
- Median: **8.7 seconds**
- Range: **7.1–17.7 seconds**
- Synthesiser usage: **80,508 tokens**
- Estimated synthesiser cost: **$0.292**
- Estimated judge cost: **$0.0018**
- Interpretation was the largest cost component at **$0.212**, about 72% of synthesiser cost.

Overall, the system is production-threshold compliant, reliable at preserving meaning, and effective at self-repair. The clearest improvement area is distinguishing genuine ambiguity from harmless unresolved pronouns, followed by more consistent categorical handling of contextual negation and statement-versus-request intent.

## Future Developments

### Hokkien speech playback for cross-checking

The elderly speaker currently has no way to confirm CareBridge heard them
correctly unless they can read the Chinese rendering. An elderly Hokkien speaker
who is not literate in Chinese has no verification path at all, which is exactly
the user we built the transcription layer for.

MERaLiON ([link]) is the candidate: a Singapore-focused speech model family
covering local languages, including Hokkien. Playing the reconciled
`cleaned_transcript` back in Hokkien would close that loop.

Three blockers kept it out of the MVP:

- It expects Hokkien-orthography input, not Mandarin characters, so the display
  layer would need a Mandarin-to-Hokkien romanisation step that does not exist yet.
- GPU dependency, which the demo machine does not have.
- A non-standard licence that needs reading before the project depends on it.

A shorter path for the demo is a small set of pre-generated Hokkien clips for the
three rehearsed demo utterances. That proves the interaction without taking on
the model.

### Persistent, access-controlled profile storage

`person_info` is currently a local JSON file (`data/profile.json`) read at
startup, plus `localStorage` on the browser side. That is fine for a seeded,
non-sensitive demo profile and it keeps the hard rule that only `PROFILE_FIELDS`
reach a prompt.

It does not survive a real deployment. Household names, relationships and
household terms are personal data about people who never consented to the
prototype. A managed store such as Supabase would give per-household row-level
access control, an audit trail, and a place to enforce retention, none of which a
JSON file on the server can offer.

The `PROFILE_FIELDS` whitelist should stay as the boundary regardless of backing
store. Clinical fields still do not belong in the profile; clinically relevant
vocabulary belongs in the glossary `ambiguity` field.

### Vector store for a larger glossary

The retriever holds its vectors in process and re-embeds anything missing from
the on-disk cache at startup. At the current glossary size this is the right call:
cold start is short, there is no external service to fail mid-demo, and alias
matching answers with no network at all.

If the glossary grows by an order of magnitude, a persistent ChromaDB collection
becomes the better trade. Cold start gets slower because the collection has to
load, but per-query search stops scanning every entry and total latency improves.
The switch should be driven by a measured regression in lookup latency, not by
entry count alone.

`MIN_SIMILARITY = 0.28` sits in a narrow band and is calibrated against a small
probe set. Any material glossary expansion needs `--calibrate` re-run before the
threshold can be trusted.

### Latency

Average end-to-end translation is roughly 9.5 s (median 8.7 s, range 7.1–17.7 s).
Interpretation is about 72% of synthesiser cost, so it is the first place to look.

The largest available saving is not in the model calls. Right now the agentic
pipeline waits for the user to press **Translate** after transcription finishes.
Firing interpretation the moment STT returns would hide several seconds of model
latency behind the time the user spends reading the transcript.

However this comes at a cost, which is a redundant run whenever users edit the transcript. The speculative run is then discarded and a second one would start, leading to wasted tokens. However, we feel it is a trade off worth taking to improve user experience.
