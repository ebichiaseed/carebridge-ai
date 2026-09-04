"""
tools/glossary_lookup.py

Hybrid Singlish/Chinese term retrieval for the Interpretation Agent.

OWNER: context lookup. Self-contained -- no dependency on the graph, the agents,
or configs.settings, so it can be built and tested before orchestration exists.

Two retrieval paths, unioned:

  1. Alias matching  -- deterministic, spacing- and punctuation-insensitive,
     word-boundary-aware match of `term` and `variants` against the transcript.
     High precision, zero cost, works with no network.

  2. Cosine similarity -- Titan v2 embeddings, exact cosine over unit vectors.
     Earns its keep on queries alias matching cannot do: paraphrases, and
     CJK-script ASR candidates matching an English-language meaning.

Alias hits always outrank vector hits. Vector hits below `min_similarity` are
dropped rather than padded in, because the interpretation prompt instructs the
model to trust these meanings -- an irrelevant hit is worse than no hit.

CONTRACT: returns list[dict], never a pre-formatted string. `lookup()` never raises.

Layout: caregiving_glossary.json sits beside this file in tools/. The resolver
also accepts data/caregiving_glossary.json at the repo root (the codebook
layout) so moving the file later needs no code change.

Usage (instantiate at app startup, not on import):

    from tools.glossary_lookup import get_retriever
    retriever = get_retriever()
    result = retriever.lookup(transcript)
    state["glossary_hits"] = result.hits

Diagnostics, no AWS needed:      python3 -m tools.glossary_lookup --check
Threshold calibration, needs AWS: python3 -m tools.glossary_lookup --calibrate
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# Configuration
TOP_K = 3
MIN_SIMILARITY = 0.35          # TODO: need to calibrate -> python3 -m tools.glossary_lookup --check, then aws sso login --profile hackathon, then python3 -m tools.glossary_lookup --calibrate
MAX_HITS = 8

MIN_ALIAS_LENGTH = 3           # latin forms
MIN_ALIAS_LENGTH_CJK = 2       # CJK characters carry more meaning per character

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMENSIONS = 1024

GLOSSARY_FILENAME = "caregiving_glossary.json"
CACHE_FILENAME = "glossary_embeddings.json"

# Read from env, NOT from configs.settings. The us.anthropic.* inference
# profiles this project uses only resolve in US regions.
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# File location

def _candidate_paths() -> list[Path]:
    """Every place the glossary might reasonably live, in priority order.

    Covers the current layout (tools/caregiving_glossary.json, i.e. beside this
    module) and the codebook layout (data/caregiving_glossary.json at the repo
    root), so relocating the file later needs no code change.
    """
    here = Path(__file__).resolve().parent
    candidates = [here / GLOSSARY_FILENAME, here / "data" / GLOSSARY_FILENAME]

    for parent in here.parents:
        candidates.append(parent / "data" / GLOSSARY_FILENAME)
        candidates.append(parent / GLOSSARY_FILENAME)
        if (parent / ".git").is_dir():
            break

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _resolve_glossary_path() -> tuple[Path | None, list[Path]]:
    """Return (found_path_or_None, paths_searched)."""
    override = os.getenv("GLOSSARY_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        return (path if path.is_file() else None), [path]

    searched = _candidate_paths()
    for path in searched:
        if path.is_file():
            return path, searched
    return None, searched


GLOSSARY_PATH, _SEARCHED_PATHS = _resolve_glossary_path()

# Keep the embedding cache beside the glossary it was built from.
_CACHE_DIR = GLOSSARY_PATH.parent if GLOSSARY_PATH else Path(__file__).resolve().parent
EMBED_CACHE_PATH = _CACHE_DIR / CACHE_FILENAME


# Text normalisation  (defined before the schema, which depends on it)

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace."""
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text.casefold())).strip()


def _squash(text: str) -> str:
    """Normalised, with all spaces removed."""
    return _normalise(text).replace(" ", "")


def _has_latin(text: str) -> bool:
    """Casefold first: without it, an uppercase-only form like "GP" or "BP"
    reads as CJK, which both lowers its length floor and strips the
    word-boundary anchors from its alias pattern."""
    return bool(re.search(r"[a-z]", text.casefold()))


def _min_alias_length(form: str) -> int:
    return MIN_ALIAS_LENGTH if _has_latin(form) else MIN_ALIAS_LENGTH_CJK


def _alias_pattern(form: str) -> re.Pattern[str]:
    r"""Spacing-insensitive matcher.

    "pai seh", "paiseh", "pai-seh" and "pai  seh" all match each other, because
    punctuation normalises to space and \s* is allowed between every character.
    Word-boundary anchors still hold at the ends, so "song" does not fire inside
    "belong" or "songkok".

    CJK gets no boundary anchors -- \b is meaningless without word separators.

    Known trade-off: \s* can span a word break, so a short form like "mai" will
    match "ma i". Raise MIN_ALIAS_LENGTH if that shows up in real traces.
    """
    body = r"\s*".join(re.escape(c) for c in _squash(form))
    if _has_latin(form):
        return re.compile(rf"(?<!\w){body}(?!\w)")
    return re.compile(body)


# Entry schema

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
        """Every spelling that should trigger an alias match, longest first.

        The length floor is script-aware: 3 characters for latin, 2 for CJK.
        A flat floor of 3 would silently drop 头晕, 跌倒, 不要, 中风 and every
        other two-character Chinese form -- roughly half this glossary.
        """
        forms = {self.term, *self.variants}
        return sorted(
            (f for f in forms if len(_squash(f)) >= _min_alias_length(f)),
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


# Loading, validation and content checks

class GlossaryConfigError(Exception):
    """Raised at construction time for a missing or malformed glossary.

    Deliberately fatal: a silent fallback to a stub glossary mid-demo is worse
    than failing at startup, where the message is visible.
    """


def _load_glossary(path: Path | None, searched: list[Path]) -> list[dict]:
    if path is None:
        listing = "\n".join(f"  - {p}" for p in searched)
        raise GlossaryConfigError(
            f"Glossary file '{GLOSSARY_FILENAME}' not found. Searched:\n{listing}\n"
            f"Fix: put the file in one of those locations, or set GLOSSARY_PATH."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GlossaryConfigError(f"Cannot read {path}: {error}") from error

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise GlossaryConfigError(
            f"{path} is not valid JSON: {error.msg} "
            f"at line {error.lineno}, column {error.colno}."
        ) from error

    if not isinstance(data, list):
        raise GlossaryConfigError(
            f"{path} must contain a JSON array of entries, got {type(data).__name__}."
        )
    return data


def _validate_entries(raw: list[dict]) -> list[GlossaryEntry]:
    """Fail at startup on a malformed glossary, not at query time.

    Reports every problem at once, so a hand-edited file can be fixed in one
    pass rather than one error per run.
    """
    entries: list[GlossaryEntry] = []
    problems: list[str] = []

    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            problems.append(f"index {index}: expected an object, got {type(row).__name__}")
            continue
        try:
            entries.append(GlossaryEntry.model_validate(row))
        except ValidationError as error:
            label = row.get("id") or row.get("term") or f"index {index}"
            problems.append(f"{label}: {error}")

    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            problems.append(f"duplicate id: {entry.id}")
        seen.add(entry.id)

    if problems:
        raise GlossaryConfigError("Invalid glossary:\n" + "\n".join(problems))
    if not entries:
        raise GlossaryConfigError("Glossary is empty.")
    return entries


def _content_warnings(entries: list[GlossaryEntry]) -> list[str]:
    """Non-fatal quality problems. Surfaced by --check and logged at startup."""
    warnings: list[str] = []

    for entry in entries:
        if not entry.surface_forms:
            longest = max((entry.term, *entry.variants), key=len, default="")
            warnings.append(
                f"{entry.id} ({entry.term}): no form long enough to alias-match "
                f"(longest is {longest!r}). Only vector search can find it. "
                f"Add a longer variant."
            )

    owners: dict[str, list[str]] = {}
    for entry in entries:
        for form in entry.surface_forms:
            owners.setdefault(_squash(form), []).append(entry.id)
    for form, ids in sorted(owners.items()):
        if len(ids) > 1:
            warnings.append(
                f"surface form {form!r} is claimed by {', '.join(ids)} -- "
                f"both entries will fire on the same text."
            )

    return warnings


# Embedder -- lazy client, no AWS work at import time

def _explain_aws_error(error: Exception) -> str:
    """Turn a botocore exception into something actionable."""
    text = f"{type(error).__name__}: {error}"
    lowered = text.lower()

    if "expired" in lowered or ("token" in lowered and "retriev" in lowered):
        return "AWS credentials expired. Run: aws sso login --profile hackathon"
    if "nocredentials" in lowered or "unable to locate credentials" in lowered:
        return "No AWS credentials found. Run: aws sso login --profile hackathon"
    if "accessdenied" in lowered or "not authorized" in lowered:
        return (
            f"Access denied for {EMBED_MODEL_ID}. Enable model access for Titan "
            f"Embeddings in the Bedrock console for region {AWS_REGION}."
        )
    if "could not connect" in lowered or "endpointconnection" in lowered:
        return f"Cannot reach Bedrock in {AWS_REGION}. Check the region and your network."
    if "throttl" in lowered or "toomanyrequests" in lowered:
        return "Bedrock is throttling. Retry, or slow down indexing."
    return text


class TitanEmbedder:
    """Titan Text Embeddings v2. `normalize=True` returns unit vectors, so a
    plain dot product IS cosine similarity."""

    def __init__(
        self,
        region: str = AWS_REGION,
        model_id: str = EMBED_MODEL_ID,
        dimensions: int = EMBED_DIMENSIONS,
    ):
        self.model_id = model_id
        self.dimensions = dimensions
        self.region = region
        self._client = None

        if not region.startswith("us-"):
            logger.warning(
                "glossary: AWS_REGION=%s is not a US region. The us.anthropic.* "
                "inference profiles used by this project will not resolve there.",
                region,
            )

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as error:
                raise RuntimeError(
                    "boto3 is not installed. Run: pip install -r requirements.txt"
                ) from error
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("cannot embed empty text")

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {"inputText": text, "dimensions": self.dimensions, "normalize": True}
            ),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())

        vector = payload.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"unexpected embedding response, keys: {list(payload)}")
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"expected a {self.dimensions}-dim embedding, got {len(vector)}"
            )
        return vector


# Result object

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
    """Exact cosine. Titan vectors are already unit-norm, but normalise
    defensively so a stale cached vector cannot skew scores."""
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
        glossary_path: Path | None = GLOSSARY_PATH,
        embedder: TitanEmbedder | None = None,
        cache_path: Path | None = EMBED_CACHE_PATH,
    ):
        if glossary is not None:
            raw = glossary
            self.source = "<in-memory>"
        else:
            raw = _load_glossary(glossary_path, _SEARCHED_PATHS)
            self.source = str(glossary_path)

        self.entries = _validate_entries(raw)

        self.warnings = _content_warnings(self.entries)
        for warning in self.warnings:
            logger.warning("glossary: %s", warning)

        self._patterns = {
            entry.id: [(form, _alias_pattern(form)) for form in entry.surface_forms]
            for entry in self.entries
        }
        self._embedder = embedder if embedder is not None else TitanEmbedder()
        self._cache_path = cache_path
        self._vectors: dict[str, list[float]] = {}

        logger.info("glossary: loaded %d entries from %s", len(self.entries), self.source)

    # indexing 

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
            except Exception as error:
                logger.warning(
                    "glossary: embedding failed at entry %s -- %s. "
                    "Continuing with alias matching only.",
                    entry.id,
                    _explain_aws_error(error),
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
        logger.info("glossary: %d entries indexed for vector search", len(vectors))
        return True

    # lookup 

    def lookup(
        self,
        query: str | Sequence[str],
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        max_hits: int = MAX_HITS,
    ) -> GlossaryLookupResult:
        """Hybrid retrieval. Accepts one transcript or several ASR candidates.

        Pass ALL candidates. If only the Whisper candidate is queried, terms
        that only Qwen heard correctly are never looked up.
        """
        started = time.perf_counter()

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be in [0, 1], got {min_similarity}")
        if max_hits < 1:
            raise ValueError(f"max_hits must be >= 1, got {max_hits}")

        texts = self._clean_candidates(query)
        if not texts:
            return GlossaryLookupResult([], True, False, 0.0)

        alias_hits = self._alias_search(" \n ".join(texts))
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
                logger.warning(
                    "glossary: vector search failed -- %s", _explain_aws_error(err)
                )

        hits = (alias_hits + vector_hits)[:max_hits]
        latency = (time.perf_counter() - started) * 1000

        # success == the tool returned a usable result. Zero hits on a
        # transcript containing no glossary terms is a success, not a failure.
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

    # diagnostics 

    def describe(self) -> dict[str, Any]:
        """Summary for --check and for a startup readiness log."""
        categories: dict[str, int] = {}
        for entry in self.entries:
            categories[entry.category] = categories.get(entry.category, 0) + 1
        return {
            "source": self.source,
            "entries": len(self.entries),
            "categories": dict(sorted(categories.items())),
            "with_ambiguity": sum(1 for e in self.entries if e.ambiguity),
            "alias_forms": sum(len(e.surface_forms) for e in self.entries),
            "vectors_indexed": len(self._vectors),
            "warnings": self.warnings,
            "region": self._embedder.region,
            "cache_path": str(self._cache_path) if self._cache_path else None,
        }


    @staticmethod
    def _clean_candidates(query: str | Sequence[str]) -> list[str]:
        """Accept a string or a candidate list; drop blanks and non-strings.

        The ASR layer can hand back None or a non-string when one model fails,
        and that must not take down retrieval for the candidate that worked.
        """
        raw = [query] if isinstance(query, str) else list(query or [])
        texts: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                logger.warning(
                    "glossary: skipping non-string candidate (%s)", type(item).__name__
                )
                continue
            if item.strip():
                texts.append(item)
        return texts

    def _alias_search(self, text: str) -> list[dict[str, Any]]:
        normalised = _normalise(text)
        scored: list[tuple[int, GlossaryEntry]] = []

        for entry in self.entries:
            best = 0
            for form, pattern in self._patterns[entry.id]:
                if pattern.search(normalised):
                    best = max(best, len(_squash(form)))
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


    def _read_cache(self) -> dict[str, list[float]]:
        """Load cached vectors, discarding anything malformed or wrong-sized.

        A corrupt cache must degrade to re-embedding, never to a crash or to
        vectors of the wrong dimension silently entering the index.
        """
        if not self._cache_path or not self._cache_path.is_file():
            return {}

        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            logger.warning(
                "glossary: unreadable embedding cache at %s (%s); re-embedding",
                self._cache_path,
                error,
            )
            return {}

        if not isinstance(data, dict):
            logger.warning("glossary: cache is not a JSON object; re-embedding")
            return {}

        expected = self._embedder.dimensions
        clean: dict[str, list[float]] = {}
        dropped = 0
        for key, vector in data.items():
            if (
                isinstance(key, str)
                and isinstance(vector, list)
                and len(vector) == expected
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vector)
            ):
                clean[key] = vector
            else:
                dropped += 1

        if dropped:
            logger.warning("glossary: dropped %d malformed cache entries", dropped)
        return clean

    def _write_cache(self, cache: dict[str, list[float]]) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._cache_path.with_suffix(".tmp")
            temp.write_text(json.dumps(cache), encoding="utf-8")
            temp.replace(self._cache_path)   # atomic: never a half-written cache
        except OSError as error:
            logger.warning(
                "glossary: could not write cache to %s (%s)", self._cache_path, error
            )


def _cache_key(entry: GlossaryEntry, embedder: TitanEmbedder) -> str:
    """Keyed on the embed text AND the model config, so changing model or
    dimensions invalidates the cache instead of mixing vector spaces."""
    payload = f"{embedder.model_id}|{embedder.dimensions}|{entry.embed_text()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Module-level convenience for the graph node

_retriever: GlossaryRetriever | None = None


def get_retriever(**kwargs) -> GlossaryRetriever:
    """Process-wide singleton. Call this once in the FastAPI startup hook --
    the first call embeds every uncached entry and must not sit inside a
    user request."""
    global _retriever
    if _retriever is None:
        _retriever = GlossaryRetriever(**kwargs)
        _retriever.index()
    return _retriever


def reset_retriever() -> None:
    """Drop the singleton. For tests, and for reloading an edited glossary."""
    global _retriever
    _retriever = None


def glossary_lookup(query: str | Sequence[str], **kwargs) -> list[dict[str, Any]]:
    """Thin wrapper for callers that only want the hits."""
    return get_retriever().lookup(query, **kwargs).hits


# CLI


_CALIBRATION_PROBES = [
    # (utterance, glossary term it should retrieve, or None for a negative)
    ("ah ma never take her morning medicine", "never"),
    ("she cannot tolerate the pain anymore", "受不了"),
    ("after standing up she feel giddy", "giddy"),
    ("her lower back is sore from standing too long", "酸"),
    ("can you bring her to the clinic tomorrow", "send"),
    ("她今天左手没力，拿不动杯子", "没力"),
    ("吃太多炸的东西会不舒服", "热气"),
    ("what time does the bus come", None),
    ("please pass me the tv remote", None),
    ("the weather today is very hot", None),
]

_ALIAS_SMOKE_TESTS = [
    ("她说她头晕", "giddy"),
    ("阿嬤在厕所跌倒了", "跌倒"),
    ("ah ma never take her medicine", "never"),
    ("不要给她蓝色的", "don't want"),
]


def _run_check() -> int:
    """Everything verifiable without AWS."""
    try:
        retriever = GlossaryRetriever()
    except GlossaryConfigError as error:
        print(f"FAIL\n{error}")
        return 1

    info = retriever.describe()
    print(f"source        {info['source']}")
    print(f"entries       {info['entries']}")
    print(f"alias forms   {info['alias_forms']}")
    print(f"ambiguity     {info['with_ambiguity']} entries carry a note")
    print(f"categories    {info['categories']}")
    print(f"region        {info['region']}")
    print(f"cache         {info['cache_path']}")

    # Prove the script-aware length floor is active.
    cjk = sorted(
        {
            form
            for entry in retriever.entries
            for form in entry.surface_forms
            if not _has_latin(form) and len(_squash(form)) == 2
        }
    )
    print(f"\n2-char CJK forms matchable: {len(cjk)}  {cjk[:8]}")
    if not cjk:
        print(
            "  WARNING: no 2-character CJK forms are matchable. Is surface_forms "
            "still using the flat MIN_ALIAS_LENGTH instead of _min_alias_length?"
        )

    if info["warnings"]:
        print(f"\n{len(info['warnings'])} content warning(s):")
        for warning in info["warnings"]:
            print(f"  - {warning}")
    else:
        print("\nno content warnings")

    print("\nalias smoke test (no network):")
    failed = 0
    for text, expected in _ALIAS_SMOKE_TESTS:
        hits = [h["term"] for h in retriever.lookup(text).hits]
        ok = expected in hits
        failed += not ok
        print(f"  {'ok  ' if ok else 'MISS'} {text!r} -> {hits}")

    print(f"\n{'PASS' if not failed else f'{failed} smoke test(s) failed'}")
    return 1 if failed else 0


def _run_calibrate() -> int:
    """Positives should cluster high, negatives low. Set MIN_SIMILARITY in the gap."""
    try:
        retriever = GlossaryRetriever()
    except GlossaryConfigError as error:
        print(f"FAIL\n{error}")
        return 1

    if not retriever.index():
        print("Vector index unavailable -- see the warning above. Cannot calibrate.")
        return 1

    for probe, expected in _CALIBRATION_PROBES:
        result = retriever.lookup(probe, min_similarity=0.0)
        print(f"\n{probe!r}  (expect: {expected or 'nothing'})")
        for hit in result.hits:
            print(f"   {hit['score']:.3f}  {hit['match_type']:6s}  {hit['term']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--check":
        sys.exit(_run_check())
    if mode == "--calibrate":
        sys.exit(_run_calibrate())
    print(f"unknown option {mode!r}; use --check or --calibrate")
    sys.exit(2)