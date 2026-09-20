#!/usr/bin/env python3
"""ARION ALPHA 1 — language identification for Persian/Arabic/English.

Per Persian-phase spec: do NOT rely on character overlap alone — use
word-level function-word scoring. If the fasttext lid.176 model is available
(Kaggle: pip install fasttext or preloaded), it is used as the primary
classifier and the heuristic as tie-breaker.

Classify: fa | ar | en | other
"""
import os
import re

# fasttext lid model path (optional; auto-detected)
_FT_MODEL_PATHS = [
    os.environ.get("LID176_PATH", ""),
    "/kaggle/working/lid.176.bin",
    "evaluation/lid.176.bin",
    "lid.176.bin",
]
_ft = None
_ft_tried = False


def _load_fasttext():
    global _ft, _ft_tried
    if _ft_tried:
        return _ft
    _ft_tried = True
    try:
        import fasttext  # noqa
        for p in _FT_MODEL_PATHS:
            if p and os.path.exists(p):
                _ft = fasttext.load_model(p)
                break
    except Exception:
        _ft = None
    return _ft


PERSIAN_ONLY_LETTERS = set("یکیگپچژ")          # ی ک گ پ چ ژ
ARABIC_ONLY_LETTERS = set("ة>>>")  # placeholder replaced below
ARABIC_ONLY_LETTERS = set("\u0629\u064A\u0643\u0624\u0626\u0631\u0630\u062F\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u0644\u0645\u0646\u0647\u0648") - set("\u0647")

AR_FUNCTION_WORDS = {
    "في", "من", "على", "هذا", "هذه", "التي", "الذي", "ذلك", "تلك", "كما",
    "بينما", "إن", "لقد", "عندما", "حيث", "إلى", "الذي", "كان", "لكن",
    "هناك", "أو", "وقد", "فقد", "ثم", "أي", "ما", "لا", "هو", "هي",
}
FA_FUNCTION_WORDS = {
    "است", "را", "برای", "که", "های", "بود", "هستند", "نیست", "خیلی",
    "بسیار", "این", "آن", "با", "تا", "اما", "هم", "یک", "شود", "شد",
    "خود", "کرد", "کند", "باشد", "چگونه", "چطور", "چه", "هر", "می\u200c",
    "نمی\u200c", "های", "هایی", "باید", "می\u200cشود", "است.", "دارد",
}


def _heuristic(text):
    if not text or not text.strip():
        return "other", 0.0
    # strip code fences and inline code (they bias toward en)
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"`[^`\n]+`", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    words = re.findall(r"[\w\u0600-\u06FF\u0750-\u077F]+", t)
    n = len(words)
    if n == 0:
        return "other", 0.0
    fa_hits = sum(1 for w in words if w in FA_FUNCTION_WORDS or w.rstrip(".") in FA_FUNCTION_WORDS)
    ar_hits = sum(1 for w in words if w in AR_FUNCTION_WORDS)
    fa_letters = sum(1 for c in t if c in PERSIAN_ONLY_LETTERS)
    ascii_letters = sum(1 for c in t if c.isascii() and c.isalpha())
    arabic_letters = sum(1 for c in t if "\u0600" <= c <= "\u06FF")

    if ascii_letters > 5 and ascii_letters > arabic_letters * 2:
        return "en", 0.8
    # Persian-only letters are a strong Persian signal
    if fa_letters >= 2 and fa_hits >= ar_hits:
        return "fa", min(1.0, 0.55 + 0.08 * fa_letters + 0.05 * fa_hits)
    if ar_hits >= 2 and fa_hits == 0 and fa_letters == 0:
        return "ar", min(1.0, 0.5 + 0.1 * ar_hits)
    if arabic_letters >= 5:
        # Arabic-script but no Persian-only letters and no Persian function words:
        # could be Arabic — decide by function words
        if fa_hits == 0 and ar_hits >= 1:
            return "ar", 0.6
        return "fa", 0.55
    if fa_hits >= 1:
        return "fa", 0.5
    return "other", 0.3


def classify(text):
    """Return (lang, confidence). lang in {fa, ar, en, other}."""
    model = _load_fasttext()
    if model is not None:
        t = re.sub(r"\s+", " ", text)[:4000].replace("\n", " ")
        labels, probs = model.predict(t)
        if labels:
            lang = labels[0].replace("__label__", "")
            conf = float(probs[0])
            if lang in ("fa", "persian"):
                lang = "fa"
            elif lang in ("ar", "ara", "arb"):
                lang = "ar"
            elif lang == "en":
                lang = "en"
            else:
                lang = "other"
            # heuristic tie-breaker for the fa/ar boundary when confidence low
            if conf < 0.75:
                h, hc = _heuristic(text)
                if h in ("fa", "ar") and h != lang and hc > conf:
                    return h, hc
            return lang, conf
    return _heuristic(text)


def classify_response(response_text, user_text):
    """Classify the language a MODEL RESPONSE is written in.

    Skips code blocks when the response mixes code + prose (answers may be
    mostly code, which must not count as 'English prose').
    """
    prose = re.sub(r"```[\s\S]*?```", " ", response_text)
    if not prose.strip():
        # response is only code — check surrounding user language
        return "code"
    return classify(prose)[0]
