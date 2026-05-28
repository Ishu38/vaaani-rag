"""Cognitive X-Ray Mode — Misconception Detection Engine.

Detects not just *that* a student is wrong, but *why* — identifying
conceptual gaps, cognitive biases, shortcut dependency, and confidence
calibration issues. The "MRI scan of mathematical thinking."
"""

from .store import init_db, CognitiveStore
from .classifier import classify_error, ErrorDiagnosis
from .fingerprint import build_fingerprint
from .detector import analyze_turn, TurnAnalysis
from .remediation import generate_remediation
