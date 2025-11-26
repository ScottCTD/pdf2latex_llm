
import re
from typing import List

def metric_exact_match(pred: str, target: str) -> float:
    """Returns 1.0 if prediction exactly matches target, 0.0 otherwise."""
    return float(pred == target)

MATH_DELIMS = [
    (r'^\s*\\\[\s*', r'\s*\\\]\s*$'),
    (r'^\s*\\\(\s*', r'\s*\\\)\s*$'),
    (r'^\s*\$\$\s*',  r'\s*\$\$\s*$'),
    (r'^\s*\$\s*',    r'\s*\$\s*$'),
]

def strip_math_delims(s: str) -> str:
    for l, r in MATH_DELIMS:
        if re.search(l, s) and re.search(r, s):
            s = re.sub(l, '', s)
            s = re.sub(r, '', s)
            break
    return s

_SPACING_CMDS = r'(\\,|\\;|\\:|\\!|\\quad|\\qquad|~)'
def normalize_latex(s: str) -> str:
    s = s.strip()
    s = strip_math_delims(s)
    s = re.sub(_SPACING_CMDS, '', s)              # drop spacing-only commands
    s = re.sub(r'\s+', ' ', s).strip()            # collapse spaces
    return s

def metric_normalized_exact_match(pred: str, target: str) -> float:
    """Returns 1.0 if prediction matches target after normalizing whitespace and removing math delimiters, 0.0 otherwise."""
    return float(normalize_latex(pred) == normalize_latex(target))

# simple LaTeX-aware tokenizer
_TOK_RE = re.compile(
    r'(\\[A-Za-z]+)|'   # \command
    r'([{}_^])|'        # braces, _, ^
    r'([0-9]+)|'        # numbers
    r'([A-Za-z]+)|'     # identifiers
    r'(\S)'             # everything else (operators, punctuation)
)

def latex_tokens(s: str) -> List[str]:
    return [m.group(0) for m in _TOK_RE.finditer(s)]

def metric_normalized_edit_similarity(gold: str, pred: str) -> float:
    """Returns similarity score (0.0 to 1.0) based on token-level edit distance between normalized predictions."""
    a = latex_tokens(normalize_latex(gold))
    b = latex_tokens(normalize_latex(pred))
    # Levenshtein distance
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            prev, dp[j] = dp[j], min(
                dp[j] + 1,       # delete
                dp[j-1] + 1,     # insert
                prev + cost      # substitute
            )
    dist = dp[-1]
    denom = max(len(a), len(b)) or 1
    return 1.0 - dist / denom
