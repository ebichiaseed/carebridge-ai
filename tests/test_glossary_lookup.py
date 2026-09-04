"""
tests/test_glossary_lookup.py

Unit tests for tools/glossary_lookup.py.

No network, no AWS, no data file: every test injects an explicit glossary and a
FakeEmbedder. Run from the repo root:

    python3 -m unittest tests.test_glossary_lookup -v
    python3 -m unittest discover -s tests -v
"""

import asyncio
import hashlib
import json
import logging
import math
import re
import tempfile
import unittest
from pathlib import Path

from tools.glossary_lookup import (
    GlossaryConfigError,
    GlossaryEntry,
    GlossaryRetriever,
    _cache_key,
    _content_warnings,
    _load_glossary,
    _validate_entries,
    get_retriever,
    reset_retriever,
)


# Test doubles
DIM = 64
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
logging.getLogger("tools.glossary_lookup").setLevel(logging.CRITICAL)

class FakeEmbedder:
    """Deterministic bag-of-tokens embedder.

    Hashes each token into a fixed bucket and L2-normalises, so texts sharing
    vocabulary get a high cosine and unrelated texts get ~0. Deterministic
    across processes (hashlib, not the randomised builtin hash()).
    """

    def __init__(self, fail: bool = False, region: str = "us-east-1"):
        self.model_id = "fake-embedder"
        self.dimensions = DIM
        self.region = region
        self.fail = fail
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("simulated Bedrock failure")
        if not text or not text.strip():
            raise ValueError("cannot embed empty text")

        vector = [0.0] * DIM
        for token in _TOKEN.findall(text.casefold()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def row(**overrides) -> dict:
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
    row(id="g001", term="paiseh", variants=["pai seh", "paisay", "不好意思"]),
    row(
        id="g002",
        term="受不了",
        language="Chinese/Singlish",
        meaning="cannot tolerate or cannot stand something any longer",
        variants=["忍不住", "buay tahan", "cannot tahan", "cannot stand"],
        ambiguity="pain or frustration",
    ),
    row(
        id="g003",
        term="giddy",
        language="Singapore English",
        meaning="dizzy or lightheaded",
        variants=["头晕", "dizzy"],
    ),
    row(id="g004", term="song", language="Singlish", meaning="feeling good"),
    row(
        id="g005",
        term="跌倒",
        language="Chinese",
        meaning="to fall down",
        variants=["摔倒", "fall down"],
    ),
]


def build(glossary=None, embedder=None, index=True) -> GlossaryRetriever:
    embedder = embedder or FakeEmbedder()
    retriever = GlossaryRetriever(
        glossary=glossary if glossary is not None else GLOSSARY,
        embedder=embedder,
        cache_path=None,          # never touch disk unless a test asks for it
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
            row(id="a", term="tahan", meaning="endure"),
            row(id="b", term="cannot tahan", meaning="cannot endure"),
        ]
        retriever = build(glossary=glossary, index=False)
        self.assertEqual(
            terms(retriever.lookup("she cannot tahan the pain"))[0], "cannot tahan"
        )

    def test_alias_hits_score_one_and_are_tagged(self):
        retriever = build(index=False)
        hit = retriever.lookup("wa paiseh").hits[0]
        self.assertEqual(hit["match_type"], "alias")
        self.assertEqual(hit["score"], 1.0)


class ScriptAwareLengthFloorTests(unittest.TestCase):
    """The bug this guards: a flat MIN_ALIAS_LENGTH of 3 silently drops every
    two-character Chinese form -- 头晕, 跌倒, 不要, 中风 -- from alias matching."""

    def test_two_character_cjk_form_is_matchable(self):
        model = GlossaryEntry.model_validate(row(id="c", term="跌倒", language="Chinese"))
        self.assertIn("跌倒", model.surface_forms)

    def test_two_character_cjk_actually_matches_in_a_lookup(self):
        retriever = build(index=False)
        self.assertIn("跌倒", terms(retriever.lookup("阿嬤在厕所跌倒了")))

    def test_two_character_latin_form_is_still_excluded(self):
        model = GlossaryEntry.model_validate(row(id="l", term="ok", variants=["no"]))
        self.assertEqual(model.surface_forms, [])

    def test_uppercase_latin_form_is_not_mistaken_for_cjk(self):
        """Regression: _has_latin must casefold. Otherwise "GP"/"BP" take the
        CJK branch -- 2-char floor, and no word-boundary anchors, so "gp"
        would match inside unrelated words."""
        model = GlossaryEntry.model_validate(row(id="u", term="GP", variants=["BP"]))
        self.assertEqual(model.surface_forms, [])

    def test_uppercase_variant_does_not_match_inside_a_word(self):
        retriever = build(
            glossary=[row(id="u", term="polyclinic", variants=["GP"])], index=False
        )
        self.assertEqual(terms(retriever.lookup("she was upgrading her phone")), [])

    def test_single_character_cjk_is_excluded(self):
        """风 alone would fire inside 中风 (stroke) and 风扇 (fan)."""
        model = GlossaryEntry.model_validate(
            row(id="s", term="风", language="Chinese", variants=["中风"])
        )
        self.assertNotIn("风", model.surface_forms)
        self.assertIn("中风", model.surface_forms)


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
        self.assertEqual(retriever.lookup(query, min_similarity=0.9).hits, [])

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



# Input validation

class InputValidationTests(unittest.TestCase):

    def test_bad_parameters_raise_immediately(self):
        retriever = build(index=False)
        for kwargs in (
            {"top_k": 0},
            {"min_similarity": 1.5},
            {"min_similarity": -0.1},
            {"max_hits": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    retriever.lookup("wa paiseh", **kwargs)

    def test_non_string_candidates_are_skipped_not_fatal(self):
        """A failed ASR model can hand back None. The candidate that worked
        must still be searched."""
        retriever = build(index=False)
        result = retriever.lookup([None, "wa paiseh lah", 42])
        self.assertIn("paiseh", terms(result))
        self.assertTrue(result.success)

    def test_empty_query_is_a_success_with_no_hits(self):
        retriever = build()
        for query in ("   ", [], [None], ["", "  "]):
            with self.subTest(query=query):
                result = retriever.lookup(query)
                self.assertEqual(result.hits, [])
                self.assertTrue(result.success)

    def test_no_match_is_a_success_not_a_failure(self):
        """Zero hits on a transcript with no glossary terms is correct
        behaviour, and must not count against Tool-Call Success Rate."""
        retriever = build()
        self.assertTrue(retriever.lookup("what time is the bus coming").success)


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
        trace = retriever.lookup(
            "wa paiseh, cannot tolerate anymore", min_similarity=0.2
        ).to_trace()
        self.assertEqual(trace["node"], "glossary")
        self.assertEqual(trace["matches"], trace["alias_matches"] + trace["vector_matches"])
        self.assertGreaterEqual(trace["alias_matches"], 1)
        self.assertTrue(trace["tool_success"])

    def test_describe_reports_load_state(self):
        retriever = build()
        info = retriever.describe()
        self.assertEqual(info["entries"], len(GLOSSARY))
        self.assertEqual(info["vectors_indexed"], len(GLOSSARY))
        self.assertEqual(info["warnings"], [])

    def test_async_wrapper_returns_the_same_hits(self):
        retriever = build(index=False)
        sync = retriever.lookup("wa paiseh")
        result = asyncio.run(retriever.alookup("wa paiseh"))
        self.assertEqual(terms(sync), terms(result))



# Glossary file loading and validation

class FileLoadingTests(unittest.TestCase):

    def test_missing_file_error_lists_every_path_searched(self):
        missing = Path("/nonexistent/dir/caregiving_glossary.json")
        with self.assertRaises(GlossaryConfigError) as caught:
            _load_glossary(None, [missing])
        message = str(caught.exception)
        self.assertIn(str(missing), message)
        self.assertIn("GLOSSARY_PATH", message)

    def test_malformed_json_reports_line_and_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('[{"id": "g001",}]', encoding="utf-8")
            with self.assertRaises(GlossaryConfigError) as caught:
                _load_glossary(path, [path])
            self.assertIn("line", str(caught.exception))

    def test_json_object_instead_of_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obj.json"
            path.write_text('{"g001": {}}', encoding="utf-8")
            with self.assertRaisesRegex(GlossaryConfigError, "array"):
                _load_glossary(path, [path])

    def test_a_real_file_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "good.json"
            path.write_text(json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8")
            retriever = GlossaryRetriever(
                glossary_path=path, embedder=FakeEmbedder(), cache_path=None
            )
            self.assertEqual(len(retriever.entries), len(GLOSSARY))
            self.assertIn("paiseh", terms(retriever.lookup("wa paiseh")))


class ValidationTests(unittest.TestCase):

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(GlossaryConfigError, "duplicate id"):
            _validate_entries([row(id="dup", term="one"), row(id="dup", term="two")])

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(GlossaryConfigError, "Invalid glossary"):
            _validate_entries([row(meannig="typo'd key")])

    def test_missing_required_field_is_rejected(self):
        bad = row()
        del bad["meaning"]
        with self.assertRaisesRegex(GlossaryConfigError, "Invalid glossary"):
            _validate_entries([bad])

    def test_non_object_row_is_rejected(self):
        with self.assertRaisesRegex(GlossaryConfigError, "expected an object"):
            _validate_entries(["just a string"])

    def test_empty_glossary_is_rejected(self):
        with self.assertRaisesRegex(GlossaryConfigError, "empty"):
            _validate_entries([])

    def test_every_problem_is_reported_not_just_the_first(self):
        bad = [row(id="x", meannig="typo"), row(id="y", langauge="typo")]
        with self.assertRaises(GlossaryConfigError) as caught:
            _validate_entries(bad)
        self.assertIn("x", str(caught.exception))
        self.assertIn("y", str(caught.exception))


class ContentWarningTests(unittest.TestCase):

    def test_entry_with_no_matchable_form_is_reported(self):
        entries = _validate_entries([row(id="short", term="ok", variants=["no"])])
        warnings = _content_warnings(entries)
        self.assertEqual(len(warnings), 1)
        self.assertIn("short", warnings[0])

    def test_surface_form_claimed_by_two_entries_is_reported(self):
        entries = _validate_entries(
            [
                row(id="a", term="paiseh"),
                row(id="b", term="malu", variants=["pai seh"]),
            ]
        )
        warnings = _content_warnings(entries)
        self.assertTrue(any("claimed by" in w for w in warnings))

    def test_a_clean_glossary_produces_no_warnings(self):
        self.assertEqual(_content_warnings(_validate_entries(GLOSSARY)), [])

    def test_warnings_are_exposed_on_the_retriever(self):
        retriever = build(glossary=[row(id="short", term="ok")], index=False)
        self.assertEqual(len(retriever.warnings), 1)


# Embedding cache

class CacheTests(unittest.TestCase):

    def test_key_changes_when_the_entry_text_changes(self):
        embedder = FakeEmbedder()
        a = GlossaryEntry.model_validate(row(meaning="one meaning"))
        b = GlossaryEntry.model_validate(row(meaning="another meaning"))
        self.assertNotEqual(_cache_key(a, embedder), _cache_key(b, embedder))

    def test_key_changes_when_the_model_config_changes(self):
        """Otherwise a dimension change silently mixes two vector spaces."""
        model = GlossaryEntry.model_validate(row())
        small, large = FakeEmbedder(), FakeEmbedder()
        large.dimensions = 256
        self.assertNotEqual(_cache_key(model, small), _cache_key(model, large))

    def test_cached_vectors_are_reused_on_a_second_index(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            embedder = FakeEmbedder()

            first = GlossaryRetriever(glossary=GLOSSARY, embedder=embedder, cache_path=cache)
            self.assertTrue(first.index())
            calls_after_first = len(embedder.calls)
            self.assertTrue(cache.is_file())

            second = GlossaryRetriever(glossary=GLOSSARY, embedder=embedder, cache_path=cache)
            self.assertTrue(second.index())
            self.assertEqual(len(embedder.calls), calls_after_first)   # no new calls

    def test_corrupt_cache_falls_back_to_re_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            cache.write_text("not json at all", encoding="utf-8")
            embedder = FakeEmbedder()
            retriever = GlossaryRetriever(
                glossary=GLOSSARY, embedder=embedder, cache_path=cache
            )
            self.assertTrue(retriever.index())
            self.assertEqual(len(embedder.calls), len(GLOSSARY))

    def test_cache_entries_with_wrong_dimensions_are_dropped(self):
        """A stale cache from a different embedding config must not enter the
        index with vectors of the wrong size."""
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            embedder = FakeEmbedder()
            entry = GlossaryEntry.model_validate(GLOSSARY[0])
            cache.write_text(
                json.dumps({_cache_key(entry, embedder): [0.1, 0.2, 0.3]}),
                encoding="utf-8",
            )
            retriever = GlossaryRetriever(
                glossary=GLOSSARY, embedder=embedder, cache_path=cache
            )
            self.assertTrue(retriever.index())
            self.assertEqual(len(embedder.calls), len(GLOSSARY))   # all re-embedded

    def test_cache_that_is_not_an_object_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            cache.write_text("[1, 2, 3]", encoding="utf-8")
            retriever = GlossaryRetriever(
                glossary=GLOSSARY, embedder=FakeEmbedder(), cache_path=cache
            )
            self.assertTrue(retriever.index())

    def test_no_temp_file_is_left_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            retriever = GlossaryRetriever(
                glossary=GLOSSARY, embedder=FakeEmbedder(), cache_path=cache
            )
            retriever.index()
            self.assertFalse(cache.with_suffix(".tmp").exists())


# Singleton

class SingletonTests(unittest.TestCase):

    def tearDown(self):
        reset_retriever()

    def test_get_retriever_returns_the_same_instance(self):
        first = get_retriever(glossary=GLOSSARY, embedder=FakeEmbedder(), cache_path=None)
        second = get_retriever()
        self.assertIs(first, second)

    def test_reset_allows_reloading_an_edited_glossary(self):
        first = get_retriever(glossary=GLOSSARY, embedder=FakeEmbedder(), cache_path=None)
        reset_retriever()
        second = get_retriever(
            glossary=GLOSSARY[:2], embedder=FakeEmbedder(), cache_path=None
        )
        self.assertIsNot(first, second)
        self.assertEqual(len(second.entries), 2)


if __name__ == "__main__":
    unittest.main()