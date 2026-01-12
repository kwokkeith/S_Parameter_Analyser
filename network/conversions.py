def s_to_abcd(
    z_0: float, S_11: complex, S_12: complex, S_21: complex, S_22: complex
) -> tuple[complex, complex, complex, complex]:
    """Converts S-parameters to ABCD parameters."""
    A = ((1 + S_11) * (1 - S_22) + S_12 * S_21) / (2 * S_21)
    B = ((1 + S_11) * (1 + S_22) - S_12 * S_21) / (2 * S_21) * z_0
    C = ((1 - S_11) * (1 - S_22) - S_12 * S_21) / (2 * S_21) / z_0
    D = ((1 - S_11) * (1 + S_22) + S_12 * S_21) / (2 * S_21)
    return A, B, C, D


def abcd_to_s(
    z_0: float, A: complex, B: complex, C: complex, D: complex
) -> tuple[complex, complex, complex, complex]:
    """Converts ABCD-parameters to S parameters."""
    denom = A + B / z_0 + C * z_0 + D
    S_11 = (A + B / z_0 - C * z_0 - D) / denom
    S_12 = 2 * (A * D - B * C) / denom
    S_21 = 2 / denom
    S_22 = (-A + B / z_0 - C * z_0 + D) / denom
    return S_11, S_12, S_21, S_22
