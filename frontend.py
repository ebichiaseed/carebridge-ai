"""HTTP layer for the CareBridge parallel transcription pipeline.

Run from the repository root (the same directory as this file):

    uvicorn frontend:app --reload

Both ASR models load once at startup, not per request.
"""

import asyncio
import io
import json
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from agents.interpretation_agent import InterpretationAgent
from agents.structure_agent import StructureAgent
from agents.verify_agent import VerifyAgent
from configs.settings import (
    AWS_REGION,
    INTERPRETATION_MODEL,
    POLLY_ENGINE,
    POLLY_VOICE_ID,
    TRANSLATION_MODEL,
    VERIFICATION_MODEL,
)
from models.model_factory import ModelFactory
from services.polly_service import PollyService
from services.run_log_service import write_run_log
from services.transcription_agent.transcription_service import ParallelTranscriptionService
from state import create_initial_state
from synthesiser import build_workflow
from tools.glossary_lookup import get_retriever

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ContextTurn(BaseModel):
    speaker: str
    text: str


class TranslationRequest(BaseModel):
    transcript: str | list[str]
    recent_context: list[ContextTurn] = Field(default_factory=list)
    # Omit to use the saved settings profile. Send a value only to override it
    # for this one turn; send {} to deliberately run with no person info.
    person_info: dict[str, Any] | None = None


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)


# ---------------------------------------------------------------------------
# Settings profile
#
# One local profile, since the demo is one household. The five PROFILE_FIELDS
# are exactly what the interpretation agent reads as `person_info`.
# ui_language is frontend-only and is stripped before the agent sees it.
# ---------------------------------------------------------------------------

PROFILE_FIELDS = (
    "preferred_name",
    "languages",
    "household_terms",
    "relationships",
    "communication_preferences",
)

PROFILE_PATH = Path(__file__).parent / "data" / "profile.json"


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_name: str = ""
    # e.g. ["English", "Mandarin", "Hokkien"]
    languages: list[str] = Field(default_factory=list)
    # e.g. {"the white one": "white blood pressure tablet, 8am dose"}
    household_terms: dict[str, str] = Field(default_factory=dict)
    # e.g. {"ah girl": "Siti, the domestic helper"}
    relationships: dict[str, str] = Field(default_factory=dict)
    communication_preferences: str = ""
    # Interface language only. Does not affect interpretation or translation.
    ui_language: Literal["en", "zh"] = "en"

    def person_info(self) -> dict[str, Any]:
        """The subset the interpretation agent understands."""
        return self.model_dump(include=set(PROFILE_FIELDS))


def _read_profile() -> Profile:
    try:
        raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Profile()
    except (json.JSONDecodeError, OSError):
        logger.warning("Profile at %s is unreadable; using defaults.", PROFILE_PATH)
        return Profile()

    try:
        return Profile.model_validate(raw)
    except ValidationError:
        logger.warning("Profile at %s failed validation; using defaults.", PROFILE_PATH)
        return Profile()


def _write_profile(profile: Profile) -> None:
    """Atomic write, so a crash mid-save cannot leave half a profile on disk."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_PATH.with_suffix(".tmp")
    tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(PROFILE_PATH)


# ---------------------------------------------------------------------------
# Detected language
#
# The interpretation agent already returns source_language, so the UI needs no
# input toggle. This collapses the model's free text into a small set of codes;
# the frontend owns the display wording. Dialects report as "chinese", since
# the ASR returns them as Chinese characters anyway.
# ---------------------------------------------------------------------------

_CHINESE_MARKERS = (
    "chinese", "mandarin", "hokkien", "cantonese", "teochew", "hakka",
    "minnan", "dialect", "中文", "华语", "福建", "广东", "潮州",
)
_ENGLISH_MARKERS = ("english", "singlish", "英文", "英语")
_MALAY_MARKERS = ("malay", "bahasa", "马来")


def _normalise_language(raw: str | None) -> str:
    if not raw:
        return "unknown"

    text = raw.casefold()
    matched = [
        code
        for code, markers in (
            ("chinese", _CHINESE_MARKERS),
            ("english", _ENGLISH_MARKERS),
            ("malay", _MALAY_MARKERS),
        )
        if any(marker in text for marker in markers)
    ]

    if len(matched) > 1:
        return "mixed"
    if matched:
        return matched[0]
    return "unknown"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carebridge")

# Formats ffmpeg handles. MediaRecorder in the browser produces .webm.
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4", ".aac"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading CareBridge services.")

    def build_transcription_service() -> ParallelTranscriptionService:
        return ParallelTranscriptionService(
            [
                ("singlish_whisper", ModelFactory.create("whisper-singlish")),
                ("qwen_multilingual", ModelFactory.create("qwen-asr")),
            ]
        )

    state["service"] = await asyncio.to_thread(build_transcription_service)

    state["workflow"] = build_workflow(
        interpretation_agent=InterpretationAgent(
            model=ModelFactory.create(INTERPRETATION_MODEL)
        ),
        structure_agent=StructureAgent(
            model=ModelFactory.create(TRANSLATION_MODEL)
        ),
        verify_agent=VerifyAgent(
            model=ModelFactory.create(VERIFICATION_MODEL)
        ),
    )

    state["glossary"] = get_retriever()
    state["polly"] = PollyService(AWS_REGION, POLLY_VOICE_ID, POLLY_ENGINE)
    state["profile"] = await asyncio.to_thread(_read_profile)

    logger.info("CareBridge services ready.")
    yield
    state.clear()


app = FastAPI(title="CareBridge AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"ready": "service" in state}


@app.get("/api/profile")
async def get_profile() -> dict:
    """Settings page load."""
    profile: Profile = state.get("profile") or Profile()
    return {"profile": profile.model_dump(), "fields": list(PROFILE_FIELDS)}


@app.put("/api/profile")
async def put_profile(profile: Profile) -> dict:
    """Settings page save. Replaces the whole profile, so send every field."""
    try:
        await asyncio.to_thread(_write_profile, profile)
    except OSError as error:
        logger.exception("Could not write profile")
        raise HTTPException(
            status_code=500, detail="Settings could not be saved."
        ) from error

    state["profile"] = profile
    return {"profile": profile.model_dump(), "saved": True}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    """Accept one audio file, return one candidate per ASR model."""
    service = state.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")

    suffix = Path(audio.filename or "").suffix.lower() or ".webm"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{suffix}'.",
        )

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is larger than 25 MB.")

    # The models take a path on disk, so the upload has to be written out first.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(payload)
        tmp.close()
        candidates = await _transcribe_tolerantly(service, tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    return {"candidates": candidates}


@app.post("/api/translate")
async def translate(request: TranslationRequest) -> dict:
    """Interpret, translate, and verify one or more ASR candidates.

    `transcript` is whatever the user last saw in the "what we heard" box,
    which they may have corrected, not necessarily the raw ASR output.
    """
    workflow = state.get("workflow")
    glossary = state.get("glossary")
    if workflow is None or glossary is None:
        raise HTTPException(status_code=503, detail="Translation service is still loading.")

    profile: Profile = state.get("profile") or Profile()
    person_info = (
        request.person_info
        if request.person_info is not None
        else profile.person_info()
    )
    context = [turn.model_dump() for turn in request.recent_context]

    try:
        glossary_result = await glossary.alookup(request.transcript)
        initial_state = create_initial_state(
            request.transcript,
            glossary_hits=glossary_result.hits,
            recent_context=context,
            person_info=person_info,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    run_id = str(uuid4())
    started_at = datetime.now(UTC)
    started_timer = time.perf_counter()

    try:
        result = await workflow.ainvoke(initial_state)
    except Exception as error:
        logger.exception("Translation workflow failed")
        await _log_workflow_run(
            run_id,
            request,
            started_at,
            started_timer,
            result=None,
            error=error,
        )
        raise HTTPException(
            status_code=500,
            detail="The translation could not be completed.",
        ) from error

    await _log_workflow_run(
        run_id,
        request,
        started_at,
        started_timer,
        result=result,
    )

    interpretation = result.get("interpretation")
    source_language = getattr(interpretation, "source_language", None)

    return {
        "run_id": run_id,
        "status": result.get("status", "failed"),
        "translation": result.get("final_text"),
        "clarification_question": result.get("clarification_question"),
        "detected_language": {
            "code": _normalise_language(source_language),
            "detail": source_language,
        },
        # What the pipeline settled on after reconciling the ASR candidates.
        "cleaned_transcript": getattr(interpretation, "cleaned_transcript", None),
    }


@app.post("/api/speech")
async def speech(request: SpeechRequest) -> StreamingResponse:
    """Synthesize translated text with the configured Amazon Polly voice."""
    polly = state.get("polly")
    if polly is None:
        raise HTTPException(status_code=503, detail="Speech service is still loading.")

    try:
        audio = await asyncio.to_thread(polly.synthesize, request.text)
    except Exception as error:
        logger.exception("Amazon Polly synthesis failed")
        raise HTTPException(
            status_code=502,
            detail="Amazon Polly could not generate speech. Check AWS access and configuration.",
        ) from error

    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")


async def _log_workflow_run(
    run_id: str,
    request: TranslationRequest,
    started_at: datetime,
    started_timer: float,
    *,
    result: dict | None,
    error: Exception | None = None,
) -> None:
    """Persist a workflow result without allowing logging to break the API."""

    finished_at = datetime.now(UTC)
    record = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": round((time.perf_counter() - started_timer) * 1000, 2),
        "input": request,
        "result": result,
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }

    try:
        path = await asyncio.to_thread(write_run_log, run_id, record)
        logger.info("Workflow run %s logged to %s", run_id, path)
    except Exception:
        logger.exception("Could not write workflow run log %s", run_id)


async def _transcribe_tolerantly(
    service: ParallelTranscriptionService, audio_path: str
) -> list[dict]:
    """Run every model, but let one failure through without killing the others.

    ParallelTranscriptionService.transcribe() uses a bare asyncio.gather, so a
    single model raising takes down the whole request. For a live demo it is
    better to show one transcript plus an error than to show nothing.
    """
    results = await asyncio.gather(
        *(model.transcribe(audio_path) for _, model in service.models),
        return_exceptions=True,
    )

    candidates = []
    for (name, _), result in zip(service.models, results):
        if isinstance(result, Exception):
            logger.exception("Model %s failed", name, exc_info=result)
            candidates.append({"model": name, "text": None, "error": str(result)})
        else:
            candidates.append({"model": name, "text": result.strip(), "error": None})
    return candidates


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")
