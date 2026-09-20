#!/usr/bin/env python3
"""ARION ALPHA 1 — Persian text normalization with code-aware protection.

Goal (Persian-phase spec, "PERSIAN NORMALIZATION"):
  Normalize Arabic/Persian Unicode variants so the model learns *Persian*
  orthography, WITHOUT damaging code, English identifiers, URLs, file paths,
  programming keywords or math.

What is normalized (text segments only — protected segments are left as-is):
  ي  U+064A Arabic Yeh      -> ی U+06CC Persian Yeh
  ى  U+0649 Alef Maqsura    -> ی U+06CC (Persian context)
  ك  U+0643 Arabic Kaf      -> ک U+06A9 Persian Keh
  ة  U+0629 Teh Marbuta     -> ه U+0647 (Persian orthography)
  ٱ  U+0671 Alef Wasla      -> ا U+0627
  ـ  U+0640 Tatweel         -> removed
  ٠-٩ U+0660-0669 Arabic-Indic digits -> ۰-۹ U+06F0-06F9 (Persian digits)
  U+200B/200E/200F/202A-202E/FEFF   -> removed (bidi junk / zero-width)
  control chars (except \n, \t)     -> removed
  U+FFFD                            -> kept (marks corruption; filtered upstream)
  \r\n / \r                         -> \n ; trailing spaces trimmed; 3+ blank lines -> 2

What is NEVER touched (protected segments):
  ``` fenced code blocks ```, `inline code`, URLs, e-mails, file paths with '/',
  and any segment that is mostly ASCII. ZWNJ (U+200C) is meaningful in Persian
  and is always preserved.
"""
import re

# ---------------- character maps ----------------
CHAR_MAP = {
    "\u064A": "\u06CC",  # Arabic Yeh -> Persian Yeh
    "\u0649": "\u06CC",  # Alef Maqsura -> Persian Yeh
    "\u0643": "\u06A9",  # Arabic Kaf -> Persian Keh
    "\u0629": "\u0647",  # Teh Marbuta -> Heh
    "\u0671": "\u0627",  # Alef Wasla -> Alef
    "\u0640": "",        # Tatweel -> remove
}
# Arabic-Indic digits -> Persian digits
for _i in range(10):
    CHAR_MAP[chr(0x0660 + _i)] = chr(0x06F0 + _i)

_BIDI_RE = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_CTRL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ---------------- protection ----------------
_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|$)")
_INLINE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s\"'<>]+", re.IGNORECASE)
_MAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# paths containing a slash and ASCII path-ish chars (unix or win), or ~/x
_PATH_RE = re.compile(
    r"(?:~|\.{1,2})?/[A-Za-z0-9._~+\-][A-Za-z0-9._~/+\-]*"
    r"|[A-Za-z]:\\(?:[A-Za-z0-9._~+\-]+\\?)+")
# mostly-ASCII runs (English identifiers / code-ish text), length >= 4
_ASCII_RUN_RE = re.compile(r"[A-Za-z0-9_\-+=*/<>{}()[\];:,.\"'&|!?#$%^~\\ ]{4,}")

_PROTECT_RES = [_FENCE_RE, _INLINE_RE, _URL_RE, _MAIL_RE, _PATH_RE, _ASCII_RUN_RE]


def _protect_spans(text):
    """Return list of (start, end) spans that must NOT be normalized."""
    spans = []
    taken = [False] * len(text)

    def claim(m):
        s, e = m.span()
        if not any(taken[s:e]):
            spans.append((s, e))
            for i in range(s, e):
                taken[i] = True
        return m.group(0)

    for rex in _PROTECT_RES:
        rex.sub(claim, text)
    return spans


def normalize_segment(seg):
    """Normalize one unprotected text segment."""
    out = []
    for ch in seg:
        if ch in CHAR_MAP:
            out.append(CHAR_MAP[ch])
        else:
            out.append(ch)
    s = "".join(out)
    s = _BIDI_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)          # trailing spaces
    s = re.sub(r"[ \t]{2,}", " ", s)          # collapse spaces (keep indentation single-line)
    s = re.sub(r"\n{3,}", "\n\n", s)          # 3+ blank lines -> 2
    return s


def normalize_text(text):
    """Normalize a Persian/English mixed text, protecting code/URLs/paths."""
    if not text:
        return text
    spans = _protect_spans(text)
    parts, pos = [], 0
    for s, e in spans:
        if s > pos:
            parts.append(normalize_segment(text[pos:s]))
        parts.append(text[s:e])               # protected: byte-for-byte
        pos = e
    if pos < len(text):
        parts.append(normalize_segment(text[pos:]))
    return "".join(parts)


def corruption_score(text):
    """0..1 score of broken Unicode: replacement chars, lone surrogates,
    excessive unassigned/rare symbols. >0.02 means 'suspicious'."""
    if not text:
        return 0.0
    bad = 0
    bad += text.count("\ufffd")
    bad += sum(1 for ch in text if 0xFFF0 <= ord(ch) <= 0xFFFF and ch not in "\ufffd")
    # unassigned Arabic-area blocks often indicate broken encoders
    bad += sum(1 for ch in text if 0x0700 <= ord(ch) <= 0x074F)
    return bad / max(1, len(text))


def normalize_conversation(messages):
    """Normalize every message content of a chat conversation."""
    out = []
    for m in messages:
        out.append({"role": m["role"], "content": normalize_text(m.get("content", ""))})
    return out
