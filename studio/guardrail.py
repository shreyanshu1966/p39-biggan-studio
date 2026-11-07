"""Safety guardrail (U5).

1. Input filter  - blocks prompts asking for violence, sexual content, hate, self-harm,
                   or images of real/identifiable people (deepfake-style requests).
2. Output filter - ImageNet contains weapon classes. They are allowed only for neutral
                   prompts and are always labelled; combined with harmful intent they are blocked.
3. Length limit  - long prompts are rejected (also limits prompt-injection surface).
"""
from __future__ import annotations

import re

MAX_LEN = 200

_BLOCK_PATTERNS = {
    "violence/gore": r"\b(kill(ing|ed)?|murder|blood(y)?|gore|behead|massacre|shoot(ing)? (a |the )?(person|people|man|woman|child|kid)s?|terroris[mt]|bomb(ing)? (a|the)|torture|corpse|dead body)\b",
    "sexual content": r"\b(nude|naked|nsfw|porn|sexual|sexy|explicit|topless|erotic)\b",
    "hate": r"\b(nazi|swastika|kkk|racial slur|ethnic cleansing)\b",
    "self-harm": r"\b(suicide|self[- ]harm|cutting myself|kill myself)\b",
    "real person / deepfake": r"\b(deepfake|celebrity|president|prime minister|politician|actor|actress|real person|my (ex|girlfriend|boyfriend|classmate|teacher|neighbou?r))\b",
}
_HARMFUL_INTENT = re.compile(r"\b(attack|threat|hurt|harm|school|kids?|children|crowd|target|revenge)\b", re.I)

# ImageNet-1k indices of weapon-like classes
SENSITIVE_CLASSES = {
    413: "assault rifle", 471: "cannon", 583: "guillotine", 597: "holster",
    657: "missile", 744: "projectile", 763: "revolver", 764: "rifle", 845: "syringe",
}


def check_prompt(text: str) -> tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, "Please type a description."
    if len(t) > MAX_LEN:
        return False, f"Prompt too long ({len(t)} > {MAX_LEN} characters)."
    for category, pat in _BLOCK_PATTERNS.items():
        if re.search(pat, t, re.I):
            return False, f"Blocked by safety guardrail ({category}). This studio only generates generic ImageNet objects and animals."
    return True, ""


def check_class(text: str, class_idx: int) -> tuple[bool, str]:
    """Second-stage check on the class the mapper chose."""
    if class_idx in SENSITIVE_CLASSES:
        if _HARMFUL_INTENT.search(text or ""):
            return False, f"Blocked: '{SENSITIVE_CLASSES[class_idx]}' combined with harmful-intent wording."
        return True, f"Note: '{SENSITIVE_CLASSES[class_idx]}' is a sensitive class; outputs are low-res synthetic images for education only."
    return True, ""
