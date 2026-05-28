"""Exam Pressure Simulation Mode — Adaptive Psychological Combat Training.

Simulates real JEE exam conditions: time pressure, negative marking traps,
adaptive difficulty shifts, fatigue simulation, and real-time coaching
interjections to build cognitive resilience under stress.
"""

from .store import init_db, SimulationStore
from .engine import SimulationEngine
from .pressure import PressureController, PressureConfig
from .coach import CoachInterjector
from .question_bank import QuestionBank
