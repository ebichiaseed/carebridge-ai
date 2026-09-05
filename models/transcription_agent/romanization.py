"""Transcript text normalization helpers."""

import re


_CHINESE_TEXT = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")


def romanize_chinese_text(text: str) -> str:
    """Replace Chinese characters in a transcript with tone-less Mandarin pinyin.

    Text outside Chinese character runs, including English/Singlish words and
    punctuation, is deliberately retained verbatim.
    """
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as error:
        raise RuntimeError(
            "pypinyin is missing. Run: pip install -r requirements.txt"
        ) from error

    def romanize_match(match: re.Match[str]) -> str:
        return " ".join(lazy_pinyin(match.group(), style=Style.NORMAL))

    return _CHINESE_TEXT.sub(romanize_match, text)
