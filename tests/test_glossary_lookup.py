"""Unit tests for the hybrid glossary retriever.

No network, no AWS, no data file: every test injects an explicit glossary and a
FakeEmbedder. Run with:

    python3 -m unittest tests.test_glossary_lookup -v

Adjust the import path below if the module does not live at tools/glossary_lookup.py.
"""

import asyncio
import hashlib
import math
import re
import unittest

from agents.glossary_lookup import (
    GlossaryEntry,
    GlossaryRetriever,
    _cache_key,
    _validate_entries,
)



# Test doubles

DIM = 64
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


class FakeEmbedder:
    """Deterministic bag-of-tokens embedder.

    Hashes each token into a fixed bucket and L2-normalises, so texts sharing
    vocabulary get a high cosine and unrelated texts get ~0. Deterministic
    across processes (hashlib, not the randomised builtin hash()).
    """

    def __init__(self, fail: bool = False):
        self.model_id = "fake-embedder"
        self.dimensions = DIM
        self.fail = fail
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("simulated Bedrock failure")

        vector = [0.0] * DIM
        for token in _TOKEN.findall(text.casefold()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def entry(**overrides) -> dict:
    base = {
        "id": "g001",
        "term": "paiseh",
        "language": "Singlish",
        "meaning": "embarrassed or reluctant to impose on others",
        "variants": [],
        "category": "urgency_state",
        "example": None,
        "ambiguity": None,
    }
    base.update(overrides)
    return base


GLOSSARY = [
    entry(id="g001", term="paiseh", variants=["pai seh", "paisay", "不好意思"]),
    entry(
        id="g002",
        term="受不了",
        language="Chinese/Singlish",
        meaning="cannot tolerate or cannot stand something any longer",
        variants=["忍不住", "buay tahan", "cannot tahan", "cannot stand"],
        ambiguity="pain or frustration",
    ),
    entry(
        id="g003",
        term="giddy",
        language="Singapore English",
        meaning="dizzy or lightheaded",
        variants=["头晕", "dizzy"],
    ),
    entry(
        id="g004",
        term="song",
        language="Singlish",
        meaning="feeling good or comfortable",
        variants=[],
    ),
]


def build(glossary=None, embedder=None, index=True) -> GlossaryRetriever:
    embedder = embedder or FakeEmbedder()
    retriever = GlossaryRetriever(
        glossary=glossary if glossary is not None else GLOSSARY,
        embedder=embedder,
        cache_path=None,          # never touch disk in tests
    )
    if index:
        retriever.index()
    return retriever


def terms(result) -> list[str]:
    return [hit["term"] for hit in result.hits]


# Alias matching

class AliasMatchingTests(unittest.TestCase):

    def test_spacing_and_punctuation_are_ignored(self):
        retriever = build(index=False)
        for text in ("wa paiseh lah", "wa pai seh lah", "wa pai-seh lah", "wa pai  seh lah"):
            with self.subTest(text=text):
                self.assertIn("paiseh", terms(retriever.lookup(text)))

    def test_distinct_spelling_variant_matches(self):
        retriever = build(index=False)
        self.assertIn("paiseh", terms(retriever.lookup("she very paisay to ask")))

    def test_word_boundaries_are_respected(self):
        retriever = build(index=False)
        for text in ("i belong to you", "he wore a songkok"):
            with self.subTest(text=text):
                self.assertNotIn("song", terms(retriever.lookup(text)))

    def test_boundary_does_not_block_a_real_match(self):
        retriever = build(index=False)
        self.assertIn("song", terms(retriever.lookup("after massage very song")))

    def test_cjk_matches_without_word_boundaries(self):
        retriever = build(index=False)
        self.assertIn("受不了", terms(retriever.lookup("痛到受不了了")))

    def test_cross_script_variant_matches(self):
        """A Chinese variant on an English-termed entry still fires -- this is
        how the Qwen candidate reaches an entry the Whisper candidate missed."""
        retriever = build(index=False)
        self.assertIn("giddy", terms(retriever.lookup("她说她头晕")))

    def test_longer_surface_form_ranks_first(self):
        glossary = [
            entry(id="a", term="tahan", meaning="endure"),
            entry(id="b", term="cannot tahan", meaning="cannot endure"),
        ]
        retriever = build(glossary=glossary, index=False)
        self.assertEqual(terms(retriever.lookup("she cannot tahan the pain"))[0], "cannot tahan")

    def test_alias_hits_score_one_and_are_tagged(self):
        retriever = build(index=False)
        hit = retriever.lookup("wa paiseh").hits[0]
        self.assertEqual(hit["match_type"], "alias")
        self.assertEqual(hit["score"], 1.0)


# Vector search

class VectorSearchTests(unittest.TestCase):

    def test_paraphrase_is_found_without_the_word_appearing(self):
        retriever = build()
        result = retriever.lookup("she cannot tolerate it any longer", min_similarity=0.2)
        self.assertIn("受不了", terms(result))
        hit = next(h for h in result.hits if h["term"] == "受不了")
        self.assertEqual(hit["match_type"], "vector")

    def test_threshold_drops_weak_matches(self):
        retriever = build()
        query = "what time does the bus arrive at the interchange"
        self.assertTrue(retriever.lookup(query, min_similarity=0.9).hits == [])

    def test_alias_hits_rank_above_vector_hits(self):
        retriever = build()
        result = retriever.lookup("wa paiseh, cannot tolerate it anymore", min_similarity=0.2)
        match_types = [hit["match_type"] for hit in result.hits]
        self.assertEqual(match_types, sorted(match_types, key=lambda m: m != "alias"))

    def test_alias_matched_entries_are_not_duplicated_by_vector_search(self):
        retriever = build()
        result = retriever.lookup("wa paiseh to ask for help", min_similarity=0.0)
        ids = [hit["id"] for hit in result.hits]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_candidate_is_embedded_separately(self):
        embedder = FakeEmbedder()
        retriever = build(embedder=embedder)
        embedder.calls.clear()
        retriever.lookup(["she is unwell", "她不舒服"])
        self.assertEqual(len(embedder.calls), 2)

    def test_max_hits_caps_the_result(self):
        retriever = build()
        result = retriever.lookup("paiseh 受不了 giddy song", max_hits=2, min_similarity=0.0)
        self.assertEqual(len(result.hits), 2)


# Degradation -- the tool must never take down the node

class DegradationTests(unittest.TestCase):

    def test_index_failure_leaves_alias_matching_working(self):
        retriever = build(embedder=FakeEmbedder(fail=True), index=False)
        self.assertFalse(retriever.index())

        result = retriever.lookup("wa paiseh lah")
        self.assertIn("paiseh", terms(result))
        self.assertTrue(result.success)
        self.assertFalse(result.vector_search_used)

    def test_query_time_failure_keeps_alias_hits_and_flags_the_error(self):
        embedder = FakeEmbedder()
        retriever = build(embedder=embedder)
        embedder.fail = True     # succeeded at index, now fails at query time

        result = retriever.lookup("wa paiseh lah")
        self.assertIn("paiseh", terms(result))
        self.assertFalse(result.success)
        self.assertEqual(result.error, "RuntimeError")

    def test_lookup_does_not_raise_on_a_broken_embedder(self):
        embedder = FakeEmbedder()
        retriever = build(embedder=embedder)
        embedder.fail = True
        retriever.lookup("anything at all")   # must not raise

    def test_empty_query_is_a_success_with_no_hits(self):
        retriever = build()
        result = retriever.lookup("   ")
        self.assertEqual(result.hits, [])
        self.assertTrue(result.success)

    def test_no_match_is_a_success_not_a_failure(self):
        """Zero hits on a transcript with no glossary terms is correct
        behaviour, and must not count against Tool-Call Success Rate."""
        retriever = build()
        result = retriever.lookup("what time is the bus coming")
        self.assertTrue(result.success)


# Contract and observability

class ContractTests(unittest.TestCase):

    REQUIRED_KEYS = {
        "id", "term", "language", "meaning", "example",
        "ambiguity", "category", "match_type", "score",
    }

    def test_hit_shape_matches_what_the_agent_reads(self):
        retriever = build(index=False)
        hit = retriever.lookup("wa paiseh").hits[0]
        self.assertEqual(set(hit), self.REQUIRED_KEYS)

    def test_lookup_returns_dicts_not_a_formatted_string(self):
        retriever = build(index=False)
        hits = retriever.lookup("wa paiseh").hits
        self.assertIsInstance(hits, list)
        self.assertIsInstance(hits[0], dict)

    def test_trace_counts_alias_and_vector_separately(self):
        retriever = build()
        trace = retriever.lookup("wa paiseh, cannot tolerate anymore", min_similarity=0.2).to_trace()
        self.assertEqual(trace["node"], "glossary")
        self.assertEqual(trace["matches"], trace["alias_matches"] + trace["vector_matches"])
        self.assertGreaterEqual(trace["alias_matches"], 1)
        self.assertTrue(trace["tool_success"])

    def test_async_wrapper_returns_the_same_hits(self):
        retriever = build(index=False)
        sync = retriever.lookup("wa paiseh")
        async_result = asyncio.run(retriever.alookup("wa paiseh"))
        self.assertEqual(terms(sync), terms(async_result))


# Glossary file validation

class ValidationTests(unittest.TestCase):

    def test_duplicate_ids_are_rejected(self):
        glossary = [entry(id="dup", term="one"), entry(id="dup", term="two")]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            _validate_entries(glossary)

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid glossary"):
            _validate_entries([entry(meannig="typo'd key")])

    def test_missing_required_field_is_rejected(self):
        row = entry()
        del row["meaning"]
        with self.assertRaisesRegex(ValueError, "Invalid glossary"):
            _validate_entries([row])

    def test_empty_glossary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _validate_entries([])

    def test_every_problem_is_reported_not_just_the_first(self):
        bad = [entry(id="x", meannig="typo"), entry(id="y", langauge="typo")]
        with self.assertRaises(ValueError) as caught:
            _validate_entries(bad)
        self.assertIn("x", str(caught.exception))
        self.assertIn("y", str(caught.exception))

    def test_short_forms_are_excluded_from_alias_matching(self):
        """Single-character CJK and 1-2 char latin forms would fire inside
        unrelated words, so they must not become alias patterns."""
        model = GlossaryEntry.model_validate(
            entry(id="s", term="风", variants=["中风", "ok"])
        )
        self.assertNotIn("风", model.surface_forms)
        self.assertNotIn("ok", model.surface_forms)
        self.assertIn("中风", model.surface_forms)


# Embedding cache

class CacheKeyTests(unittest.TestCase):

    def test_key_changes_when_the_entry_text_changes(self):
        embedder = FakeEmbedder()
        a = GlossaryEntry.model_validate(entry(meaning="one meaning"))
        b = GlossaryEntry.model_validate(entry(meaning="another meaning"))
        self.assertNotEqual(_cache_key(a, embedder), _cache_key(b, embedder))

    def test_key_changes_when_the_model_config_changes(self):
        """Otherwise a dimension change silently mixes two vector spaces."""
        model = GlossaryEntry.model_validate(entry())
        small = FakeEmbedder()
        large = FakeEmbedder()
        large.dimensions = 256
        self.assertNotEqual(_cache_key(model, small), _cache_key(model, large))

    def test_unchanged_entries_are_not_re_embedded(self):
        embedder = FakeEmbedder()
        retriever = GlossaryRetriever(glossary=GLOSSARY, embedder=embedder, cache_path=None)
        retriever.index()
        first = len(embedder.calls)
        retriever.index()   # cache_path=None, so this re-embeds; with a cache it would not
        self.assertEqual(len(embedder.calls), first * 2)


if __name__ == "__main__":
    unittest.main()
