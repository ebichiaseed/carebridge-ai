"""HTTP layer for the CareBridge parallel transcription pipeline.

Run from the backend/ directory (same place transcribe_parallel.py lives):

    uvicorn frontend:app --reload

Both ASR models load once at startup, not per request.
"""

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from configs.settings import QWEN_ASR_MODEL, WHISPER_MODEL
from models.qwen_asr_model import QwenAsrModel
from models.whisper_model import WhisperSinglishModel
from services.transcription_service import ParallelTranscriptionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carebridge")

# Formats ffmpeg handles. MediaRecorder in the browser produces .webm.
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4", ".aac"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both ASR models once. First run downloads weights, so this is slow."""
    logger.info("Loading ASR models. First run downloads weights from HuggingFace.")

    def build() -> ParallelTranscriptionService:
        return ParallelTranscriptionService(
            [
                ("singlish_whisper", WhisperSinglishModel(WHISPER_MODEL)),
                ("qwen_multilingual", QwenAsrModel(QWEN_ASR_MODEL)),
            ]
        )

    # load() blocks, so keep it off the event loop
    state["service"] = await asyncio.to_thread(build)
    logger.info("Models ready.")
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
