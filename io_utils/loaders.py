from network import Component
import numpy as np
import csv
from pathlib import Path


def load_component(
    comp_data: dict,
    start_hz: int,
    stop_hz: int,
    path_to_S_parameters_fdr: Path = Path(),
) -> Component:
    S_param_dict: dict[int, np.ndarray] = {}
    S_param_file = path_to_S_parameters_fdr / f"{comp_data['type']}.csv"

    with open(S_param_file, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required = {
            "Hz",
            "S11_A",
            "S11_p",
            "S21_A",
            "S21_p",
            "S12_A",
            "S12_p",
            "S22_A",
            "S22_p",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{S_param_file} must contain columns: {', '.join(sorted(required))}"
            )

        # Function converts amplitude and degree phase into complex representation
        def form_complex(S_A: str, S_p: str) -> complex:
            mag = float(row[S_A])
            phase = np.deg2rad(float(row[S_p]))
            return complex(mag * np.cos(phase), mag * np.sin(phase))

        for row in reader:
            f = int(float(row["Hz"]))

            # Match only requested freqs (within start and stop range)
            if f < start_hz or f > stop_hz:
                continue

            S11 = form_complex("S11_A", "S11_p")
            S12 = form_complex("S12_A", "S12_p")
            S21 = form_complex("S21_A", "S21_p")
            S22 = form_complex("S22_A", "S22_p")

            S_param_dict[f] = np.array([[S11, S12], [S21, S22]], dtype=np.complex128)

    # Output warning if no frequencies were found in the requested range
    if not S_param_dict:
        print(
            f"Warning: {S_param_file.name} has no rows between "
            f"{start_hz} Hz and {stop_hz} Hz"
        )

    return Component(
        name=comp_data["name"],
        type=comp_data["type"],
        S_parameter=S_param_dict,
    )
