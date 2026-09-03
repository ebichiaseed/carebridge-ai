"""
glossary_lookup: hybrid Singlish term retrieval for the Interpretation Agent.

Two retrieval paths, unioned:

  1. Alias matching  -- deterministic, word-boundary-aware match of `term` and
     `variants` against the transcript. High precision, zero cost, works with no
     network. This catches the case that matters most: the dialect word is
     literally present in the ASR output.

  2. Cosine similarity -- Titan v2 embeddings, exact cosine over unit vectors.
     Earns its keep on the queries alias matching *cannot* do: paraphrases
     ("cannot take it anymore" -> buay tahan), and CJK-script ASR candidates
     matching an English-language meaning.

Alias hits always outrank vector hits. Vector hits below `min_similarity` are
dropped rather than padded in, because the interpretation prompt instructs the
model to trust these meanings -- an irrelevant hit is worse than no hit.

Contract: returns list[dict]. The *agent* owns prompt formatting (see
InterpretationAgent._format_hits). This module never returns a pre-formatted
string and never raises out of `lookup()`.

Usage (instantiate at app startup, not on import):

    retriever = GlossaryRetriever()
    retriever.index()                       # embeds; safe if Bedrock is down
    result = retriever.lookup(transcript)   # or await retriever.alookup(...)
    state["glossary_hits"] = result.hits
    trace.emit({**result.to_trace(), "run_id": run_id})
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# Configuration

TOP_K = 3
MIN_SIMILARITY = 0.35          # tune against real utterances; see __main__
MAX_HITS = 8                   # matches the docs' retrieve_terms(limit=8)
MIN_ALIAS_LENGTH = 3           # refuse to alias-match on 1-2 char forms

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMENSIONS = 1024

_DATA_DIR = Path(__file__).resolve().parent / "data"
GLOSSARY_PATH = _DATA_DIR / "caregiving_glossary.json"
EMBED_CACHE_PATH = _DATA_DIR / "glossary_embeddings.json"


# Entry schema -- validates the hand-edited JSON file at startup

class GlossaryEntry(BaseModel):
    """One glossary row. `extra="forbid"` so a typo'd key fails loudly."""

    model_config = ConfigDict(extra="forbid")

    id: str
    term: str
    language: str
    meaning: str
    variants: list[str] = Field(default_factory=list)
    category: str = "general"
    example: str | None = None
    ambiguity: str | None = None

    @property
    def surface_forms(self) -> list[str]:
        """Every spelling that should trigger an alias match, longest first."""
        forms = {self.term, *self.variants}
        return sorted(
            (f for f in forms if len(f.strip()) >= MIN_ALIAS_LENGTH),
            key=len,
            reverse=True,
        )

    def embed_text(self) -> str:
        """Text sent to the embedder. Variants included so mangled ASR
        spellings sit near the canonical term in vector space."""
        parts = [self.term]
        if self.variants:
            parts.append(", ".join(self.variants))
        parts.append(self.meaning)
        if self.example:
            parts.append(self.example)
        return ". ".join(parts)

    def to_hit(self, match_type: str, score: float) -> dict[str, Any]:
        """Shape consumed by InterpretationAgent._format_hits."""
        return {
            "id": self.id,
            "term": self.term,
            "language": self.language,
            "meaning": self.meaning,
            "example": self.example,
            "ambiguity": self.ambiguity,
            "category": self.category,
            "match_type": match_type,   # "alias" | "vector"
            "score": round(score, 4),
        }


# Text normalisation

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace.

    Punctuation removal is what makes "pai-seh" match the variant "pai seh".
    """
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text.casefold())).strip()


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[a-z]", text))


def _alias_pattern(form: str) -> re.Pattern[str]:
    """Word-boundary regex for latin forms, plain substring for CJK.

    Without \\b, the form "song" matches "belong" and "song bo" fires on
    unrelated audio. CJK has no word boundaries, so \\b does not apply there.
    """
    escaped = re.escape(_normalise(form)).replace(r"\ ", r"\s+")
    if _has_latin(form):
        return re.compile(rf"(?<!\w){escaped}(?!\w)")
    return re.compile(escaped)


# Embedder -- lazy client, no AWS work at import time

class TitanEmbedder:
    """Titan Text Embeddings v2. `normalize=True` returns unit vectors, so a
    plain dot product *is* cosine similarity."""

    def __init__(
        self,
        region: str | None = None,
        model_id: str = EMBED_MODEL_ID,
        dimensions: int = EMBED_DIMENSIONS,
    ):
        self.model_id = model_id
        self.dimensions = dimensions
        self._region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # imported here so `import glossary_lookup` needs no boto3

            region = self._region
            if region is None:
                from configs.settings import AWS_REGION

                region = AWS_REGION
            self._client = boto3.client("bedrock-runtime", region_name=region)
        return self._client

    def embed(self, text: str) -> list[float]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": True,
                }
            ),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]


# Result object - carries observability fields

@dataclass(frozen=True)
class GlossaryLookupResult:
    hits: list[dict[str, Any]]
    success: bool
    vector_search_used: bool
    latency_ms: float
    error: str | None = None

    def to_trace(self) -> dict[str, Any]:
        """Fields for the structured run log (observability plan s.2)."""
        return {
            "node": "glossary",
            "tool_success": self.success,
            "matches": len(self.hits),
            "alias_matches": sum(1 for h in self.hits if h["match_type"] == "alias"),
            "vector_matches": sum(1 for h in self.hits if h["match_type"] == "vector"),
            "vector_search_used": self.vector_search_used,
            "latency_ms": round(self.latency_ms, 1),
            "error_code": self.error,
        }


# Retriever

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact cosine. Vectors are already unit-norm from Titan, but normalise
    defensively so a cached vector from a different config cannot skew scores."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class GlossaryRetriever:
    def __init__(
        self,
        glossary: list[dict] | None = None,
        glossary_path: Path = GLOSSARY_PATH,
        embedder: TitanEmbedder | None = None,
        cache_path: Path | None = EMBED_CACHE_PATH,
    ):
        raw = glossary if glossary is not None else _load_glossary_file(glossary_path)
        self.entries = _validate_entries(raw)
        self._patterns = {
            entry.id: [(form, _alias_pattern(form)) for form in entry.surface_forms]
            for entry in self.entries
        }
        self._embedder = embedder if embedder is not None else TitanEmbedder()
        self._cache_path = cache_path
        self._vectors: dict[str, list[float]] = {}

    def index(self) -> bool:
        """Embed every entry, using the on-disk cache for unchanged rows.

        Returns True if the vector index is usable. Never raises: alias
        matching still works with no embeddings at all.
        """
        cache = self._read_cache()
        vectors: dict[str, list[float]] = {}
        called = 0

        for entry in self.entries:
            key = _cache_key(entry, self._embedder)
            if key in cache:
                vectors[entry.id] = cache[key]
                continue
            try:
                vector = self._embedder.embed(entry.embed_text())
            except Exception as error:  # throttle, expired SSO, no access
                logger.warning(
                    "glossary: embedding failed for %s (%s); "
                    "continuing with alias matching only",
                    entry.id,
                    type(error).__name__,
                )
                self._vectors = {}
                return False
            called += 1
            vectors[entry.id] = vector
            cache[key] = vector

        self._vectors = vectors
        if called:
            self._write_cache(cache)
            logger.info("glossary: embedded %d new/changed entries", called)
        logger.info("glossary: %d entries indexed", len(vectors))
        return True


    def lookup(
        self,
        query: str | Sequence[str],
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        max_hits: int = MAX_HITS,
    ) -> GlossaryLookupResult:
        """Hybrid retrieval. Accepts one transcript or several ASR candidates.

        Passing all candidates matters: if only the Whisper candidate is
        queried, terms that only Qwen heard correctly are never looked up.
        """
        started = time.perf_counter()
        texts = [query] if isinstance(query, str) else list(query)
        joined = " \n ".join(t for t in texts if t)

        if not joined.strip():
            return GlossaryLookupResult([], True, False, 0.0)

        alias_hits = self._alias_search(joined)
        matched_ids = {h["id"] for h in alias_hits}

        vector_hits: list[dict[str, Any]] = []
        vector_used = False
        error: str | None = None

        if self._vectors:
            try:
                vector_hits = self._vector_search(
                    texts, top_k, min_similarity, exclude=matched_ids
                )
                vector_used = True
            except Exception as err:
                error = type(err).__name__
                logger.warning("glossary: vector search failed (%s)", error)

        hits = (alias_hits + vector_hits)[:max_hits]
        latency = (time.perf_counter() - started) * 1000

        # success == the tool returned a usable result. An empty result from a
        # transcript with no glossary terms in it is a success, not a failure.
        return GlossaryLookupResult(
            hits=hits,
            success=error is None,
            vector_search_used=vector_used,
            latency_ms=latency,
            error=error,
        )

    async def alookup(self, query: str | Sequence[str], **kwargs) -> GlossaryLookupResult:
        """Async wrapper. `lookup` does blocking network I/O when the vector
        path runs, so it must not be called directly from an async node."""
        return await asyncio.to_thread(self.lookup, query, **kwargs)

    # internals 

    def _alias_search(self, text: str) -> list[dict[str, Any]]:
        normalised = _normalise(text)
        scored: list[tuple[int, GlossaryEntry]] = []

        for entry in self.entries:
            best = 0
            for form, pattern in self._patterns[entry.id]:
                if pattern.search(normalised):
                    best = max(best, len(form))
            if best:
                scored.append((best, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        # score 1.0: an exact surface match is as certain as this tool gets
        return [entry.to_hit("alias", 1.0) for _, entry in scored]

    def _vector_search(
        self,
        texts: Sequence[str],
        top_k: int,
        min_similarity: float,
        exclude: set[str],
    ) -> list[dict[str, Any]]:
        by_id = {entry.id: entry for entry in self.entries}
        best: dict[str, float] = {}

        # Embed each candidate separately. A mean-pooled vector of two
        # different-language candidates points somewhere between them and can
        # match neither cleanly.
        for text in texts:
            if not text.strip():
                continue
            query_vector = self._embedder.embed(text)
            for entry_id, entry_vector in self._vectors.items():
                if entry_id in exclude:
                    continue
                score = _cosine(query_vector, entry_vector)
                if score > best.get(entry_id, 0.0):
                    best[entry_id] = score

        ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        return [
            by_id[entry_id].to_hit("vector", score)
            for entry_id, score in ranked[:top_k]
            if score >= min_similarity
        ]

    # cache 

    def _read_cache(self) -> dict[str, list[float]]:
        if not self._cache_path or not self._cache_path.is_file():
            return {}
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("glossary: unreadable embedding cache (%s)", error)
            return {}

    def _write_cache(self, cache: dict[str, list[float]]) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError as error:
            logger.warning("glossary: could not write embedding cache (%s)", error)


def _cache_key(entry: GlossaryEntry, embedder: TitanEmbedder) -> str:
    """Keyed on the embed text *and* the model config, so changing model or
    dimensions invalidates the cache instead of mixing vector spaces."""
    payload = f"{embedder.model_id}|{embedder.dimensions}|{entry.embed_text()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



# Loading and validation

def _load_glossary_file(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Glossary not found at {path}. Create it, or pass glossary=[...] "
            f"explicitly (tests should do the latter)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of entries.")
    return data


def _validate_entries(raw: list[dict]) -> list[GlossaryEntry]:
    """Fail at startup on a malformed glossary, not at query time."""
    entries: list[GlossaryEntry] = []
    problems: list[str] = []

    for index, row in enumerate(raw):
        try:
            entries.append(GlossaryEntry.model_validate(row))
        except ValidationError as error:
            label = row.get("id") or row.get("term") or f"index {index}"
            problems.append(f"{label}: {error.error_count()} problem(s) -- {error}")

    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            problems.append(f"duplicate id: {entry.id}")
        seen.add(entry.id)

    if problems:
        raise ValueError("Invalid glossary:\n" + "\n".join(problems))
    if not entries:
        raise ValueError("Glossary is empty.")
    return entries


_retriever: GlossaryRetriever | None = None


def get_retriever(**kwargs) -> GlossaryRetriever:
    """Process-wide singleton. Call once at startup so indexing happens once."""
    global _retriever
    if _retriever is None:
        _retriever = GlossaryRetriever(**kwargs)
        _retriever.index()
    return _retriever


def glossary_lookup(query: str | Sequence[str], **kwargs) -> list[dict[str, Any]]:
    """Thin wrapper for callers that only want the hits."""
    return get_retriever().lookup(query, **kwargs).hits

#run to calculate threshold
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Positives should score high, negatives should score low. Read the gap,
    # then set MIN_SIMILARITY between the two clusters. Do not guess it.
    PROBES = [
        ("ah ma paiseh to ask for help lah", "paiseh"),
        ("the pain until cannot stand already", "buay tahan"),
        ("she already ate her lunch", "makan"),
        ("阿嬤講伊足艱苦", "buay tahan"),
        ("what time is the bus coming", None),          # negative
        ("please pass me the remote control", None),    # negative
    ]

    retriever = get_retriever()
    for probe, expected in PROBES:
        result = retriever.lookup(probe, min_similarity=0.0)
        print(f"\n{probe!r}  (expect: {expected or 'nothing'})")
        for hit in result.hits:
            print(f"   {hit['score']:.3f}  {hit['match_type']:6s}  {hit['term']}")
        print(f"   trace: {result.to_trace()}")