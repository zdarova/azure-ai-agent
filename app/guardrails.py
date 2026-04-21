"""Guardrails - input validation for prompt injection, PII, and abuse."""

import re

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+",
    r"system\s*prompt",
    r"jailbreak",
    r"DAN\s+mode",
    r"pretend\s+you\s+are",
    r"<\s*script",
]

# Basic PII patterns
PII_PATTERNS = [
    (r"\b[A-Z]{2}\d{7}\b", "passport"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b", "codice_fiscale"),
    (r"\b\d{16}\b", "credit_card"),
]


def check_input(query: str) -> dict:
    """Returns {"safe": bool, "reason": str, "pii_detected": list}."""
    lower = query.lower()

    # Check prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return {"safe": False, "reason": "Potenziale prompt injection rilevato.", "pii_detected": []}

    # Check PII
    pii_found = []
    for pattern, pii_type in PII_PATTERNS:
        if re.search(pattern, query):
            pii_found.append(pii_type)

    # Check excessive length (abuse)
    if len(query) > 5000:
        return {"safe": False, "reason": "Query troppo lunga (max 5000 caratteri).", "pii_detected": pii_found}

    return {"safe": True, "reason": "", "pii_detected": pii_found}
