"""
Functions for representing hunger in animal models.
"""


def linear_hunger(
    t: int, r_n: int, h_0: int, a: float, b: float, penalty: float = 0
) -> float:
    """Linear hunger function of the form h_0 + at - br_n"""
    return h_0 + a * t - b * r_n + penalty
