from enum import Enum, auto
from dataclasses import dataclass


class TrialType(Enum):
    FI = auto()
    Peak = auto


@dataclass
class TrainingStep:
    session_number: int = 0
    trial_number: int = 0
    trial_type: TrialType = TrialType.FI
