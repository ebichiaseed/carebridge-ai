import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from models.transcription_agent.romanization import romanize_chinese_text


class RomanizationTests(unittest.TestCase):
    def test_romanizes_chinese_runs_and_preserves_other_text(self):
        pypinyin = ModuleType("pypinyin")
        pypinyin.Style = Mock(NORMAL="normal")
        pypinyin.lazy_pinyin = Mock(
            side_effect=[["wo", "jin", "tian"], ["ni", "hao"]]
        )

        with patch.dict(sys.modules, {"pypinyin": pypinyin}):
            result = romanize_chinese_text("我今天 okay lah，你好!")

        self.assertEqual(result, "wo jin tian okay lah，ni hao!")
        self.assertEqual(pypinyin.lazy_pinyin.call_count, 2)
