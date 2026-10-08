'''
planer: combine component scores to a confidence score and verdict (include/exclude)
'''

# Import internal planer objects
from planer.utils.schema import Scores, Severity, Verdict

# THRESHOLDS: confidence cutoff per severity
THRESHOLDS: dict[Severity, float] = {
    Severity.LENIENT: 0.85,
    Severity.MODERATE: 0.70,
    Severity.STRICT: 0.55,
}

# combine: calculate the geometric mean of all three scores
def combine(scores: Scores) -> float:
    return (scores.flatness * scores.orientation * scores.proximity) ** (1 / 3)

# resolve_threshold: map a severity level or an explicit cutoff
def resolve_threshold(severity: Severity | float) -> float:
    if isinstance(severity, Severity):
        return THRESHOLDS[severity]
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f'threshold must be within 0-1, got {severity!r}')
    return float(severity)

# judge: return a Verdict for a given set of scores and severity level
def judge(scores: Scores, severity: Severity | float) -> Verdict:
    confidence = combine(scores)
    return Verdict(flagged=confidence >= resolve_threshold(severity), confidence=confidence)
