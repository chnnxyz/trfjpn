from dataclasses import dataclass


@dataclass
class Bounds:
    """Boundary class used for colission detection"""

    x: tuple[float, float]
    y: tuple[float, float]
