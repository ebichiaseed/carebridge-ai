"""HTTP layer for the CareBridge parallel transcription pipeline.

Run from the repository root (the same directory as this file):

    uvicorn frontend:app --reload

Both ASR models load once at startup, not per request.
"""

import asyncio
import io
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
    QWEN_ASR_MODEL,
    TRANSLATION_MODEL,
    VERIFICATION_MODEL,
    WHISPER_MODEL,
)
from models.model_factory import ModelFactory
from models.qwen_asr_model import QwenAsrModel
from models.whisper_model import WhisperSinglishModel
from services.transcription_service import ParallelTranscriptionService
from services.polly_service import PollyService
from services.run_log_service import write_run_log
from state import create_initial_state
from synthesiser import build_workflow
from tools.glossary_lookup import get_retriever

from pydantic import BaseModel, Field


class ContextTurn(BaseModel):
    speaker: str
    text: str


class TranslationRequest(BaseModel):
    transcript: str | list[str]
    recent_context: list[ContextTurn] = Field(default_factory=list)
    person_info: dict[str, Any] = Field(default_factory=dict)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)

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
                ("singlish_whisper", WhisperSinglishModel(WHISPER_MODEL)),
                ("qwen_multilingual", QwenAsrModel(QWEN_ASR_MODEL)),
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
    """Interpret, translate, and verify one or more ASR candidates."""
    workflow = state.get("workflow")
    glossary = state.get("glossary")
    if workflow is None or glossary is None:
        raise HTTPException(status_code=503, detail="Translation service is still loading.")

    context = [turn.model_dump() for turn in request.recent_context]

    try:
        glossary_result = await glossary.alookup(request.transcript)
        initial_state = create_initial_state(
            request.transcript,
            glossary_hits=glossary_result.hits,
            recent_context=context,
            person_info=request.person_info,
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

    return {
        "run_id": run_id,
        "status": result.get("status", "failed"),
        "translation": result.get("final_text"),
        "clarification_question": result.get("clarification_question"),
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
