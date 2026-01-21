from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .network import Network


def _il_db_from_s21(s21: complex) -> float:
    """
    Insertion loss (dB) from complex S21.
    IL = -20 log10(|S21|)
    """
    mag = abs(s21)
    if mag <= 0.0:
        return float("inf")
    return -20.0 * float(np.log10(mag))


@dataclass(frozen=True)
class ComponentILResult:
    name: str
    index: int
    il_total_db: np.ndarray  # IL of full cascade over freqs
    il_without_db: np.ndarray  # IL of cascade with component removed
    delta_il_db: np.ndarray  # contribution = total - without


@dataclass(frozen=True)
class ComponentPhaseResult:
    name: str
    index: int
    coeffs_total: (
        np.ndarray
    )  # Full network coefficients [cubic, quadratic, linear, constant]
    coeffs_without: np.ndarray  # Coefficients without this component
    delta_coeffs: np.ndarray  # Contribution = total - without


class NetworkAnalyser:
    def __init__(self, network: Network):
        self.network = network

    def _total_insertion_loss_db(self) -> np.ndarray:
        """
        Returns insertion loss of the full cascaded network over self.network.freqs.
        Output array aligns with self.network.freqs ordering.
        """
        il_ls: list[float] = []
        for freq in self.network.freqs:
            s21 = self.network.S_matrix[freq][1][0]  # Cascaded S21 coefficient
            il_ls.append(_il_db_from_s21(s21))
        return np.array(il_ls, dtype=float)

    def _component_standalone_insertion_loss_db(self) -> dict[str, np.ndarray]:
        """
        Standalone insertion loss of each component (ignores cascade interactions).
        Returns dict: component name -> IL array over network.freqs.
        """
        out: dict[str, list[float]] = {}
        for comp in self.network._components:
            vals: list[float] = []
            for f in self.network.freqs:
                s21 = comp.S_parameter[f][1, 0]  # Component S21
                vals.append(_il_db_from_s21(s21))
            out[comp.name] = vals
        return {k: np.array(v, dtype=float) for k, v in out.items()}

    def _component_effective_insertion_loss_db(self) -> list[ComponentILResult]:
        """
        Effective insertion loss contribution of each component inside the cascade,
        computed via leave one out:

        delta_il_k(f) = IL_total(f) - IL_without_k(f)

        Positive delta means the component increases insertion loss at that frequency.
        """
        il_total = self._total_insertion_loss_db()
        results: list[ComponentILResult] = []

        # Rebuild a new Network each time to avoid mutating the original object.
        comps = self.network._components
        freqs = self.network.freqs
        z0 = self.network.z_0

        for idx, comp in enumerate(comps):
            reduced = comps[:idx] + comps[idx + 1 :]
            reduced_net = Network(components=reduced, freqs=freqs, z_0=z0)

            il_without: list[float] = []
            for f in freqs:
                s21 = reduced_net.S_matrix[f][1, 0]
                il_without.append(_il_db_from_s21(s21))
            il_without_arr = np.array(il_without, dtype=float)

            results.append(
                ComponentILResult(
                    name=comp.name,
                    index=idx,
                    il_total_db=il_total,
                    il_without_db=il_without_arr,
                    delta_il_db=il_total - il_without_arr,
                )
            )

        return results

    def _rank_components_by_effective_il(
        self,
        agg: str = "mean",
        f_min: Optional[int] = None,
        f_max: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Produces a ranking by an aggregate of delta IL over a frequency band.
        agg options: "mean", "rms", "max"
        """
        results = self._component_effective_insertion_loss_db()

        freqs = np.array(self.network.freqs, dtype=float)
        band_mask = np.ones_like(freqs, dtype=bool)
        if f_min is not None:
            band_mask &= freqs >= float(f_min)
        if f_max is not None:
            band_mask &= freqs <= float(f_max)

        ranking: list[tuple[str, float]] = []
        for r in results:
            x = r.delta_il_db[band_mask]
            if x.size == 0:
                score = float("nan")
            elif agg == "mean":
                score = float(np.mean(x))
            elif agg == "rms":
                score = float(np.sqrt(np.mean(x * x)))
            elif agg == "max":
                score = float(np.max(x))
            else:
                raise ValueError("agg must be one of: mean, rms, max")
            ranking.append((r.name, score))

        # Sort descending: biggest effective loss contribution first
        ranking.sort(
            key=lambda t: (float("-inf") if np.isnan(t[1]) else t[1]), reverse=True
        )
        return ranking

    @property
    def network_coefficient(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the coefficient of the network S-parameters up to degree 3 using polyfit.
        This represents the different order of effects in the network.

        Returns:
            tuple containing:
            - freqs: frequency array (Hz)
            - phase: unwrapped phase of S21 (radians)
            - coeffs: polynomial coefficients [cubic, quadratic, linear, constant]
        """
        freqs = np.array(self.network.freqs, dtype=float)

        # Extract S21 phase for each frequency
        phase_values: list[float] = []
        for freq in self.network.freqs:
            s21 = self.network.S_matrix[freq][1][0]  # Cascaded S21 coefficient
            phase = np.angle(s21)  # Phase in radians
            phase_values.append(phase)

        phase = np.array(phase_values, dtype=float)

        # Unwrap phase to remove discontinuities at +-pi
        phase_unwrapped = np.unwrap(phase)

        # Normalise frequency
        freq_normalized = (freqs - freqs.mean()) / freqs.std()

        # Fit polynomial of degree 3: phase = c3*f³ + c2*f² + c1*f + c0
        # Returns coefficients in descending order: [c3, c2, c1, c0]
        coeffs = np.polyfit(freq_normalized, phase_unwrapped, deg=3)

        return freqs, phase_unwrapped, coeffs

    def _evaluate_phase_polynomial(
        self, coeffs: np.ndarray, freqs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evaluate the phase polynomial at given frequencies.

        Args:
            coeffs: Polynomial coefficients, a b c d in [a f^3, b f^2, c f, d]
            freqs: Frequencies to evaluate at (Hz). If None, uses network.freqs

        Returns:
            Predicted phase values (radians)
        """
        if freqs is None:
            freqs = np.array(self.network.freqs, dtype=float)

        # same normalisation as get_network_coefficient
        network_freqs = np.array(self.network.freqs, dtype=float)
        freq_mean = network_freqs.mean()
        freq_std = network_freqs.std()
        freq_normalised = (freqs - freq_mean) / freq_std

        return np.polyval(coeffs, freq_normalised)

    def _component_phase_error_contribution(self) -> list[ComponentPhaseResult]:
        """
        Compute the phase error contribution of each component by removing it
        from the cascade and comparing phase coefficients.

        Returns:
            List of ComponentPhaseResult for each component
        """
        # Get full network phase coefficients
        _, _, coeffs_total = self.network_coefficient

        results: list[ComponentPhaseResult] = []
        comps = self.network._components
        freqs = self.network.freqs
        z0 = self.network.z_0

        for idx, comp in enumerate(comps):
            # Create network without this component
            reduced = comps[:idx] + comps[idx + 1 :]
            reduced_net = Network(components=reduced, freqs=freqs, z_0=z0)
            reduced_analyser = NetworkAnalyser(reduced_net)

            # Get phase coefficients without this component
            _, _, coeffs_without = reduced_analyser.network_coefficient

            # Delta = total - without (contribution of this component)
            delta_coeffs = coeffs_total - coeffs_without

            results.append(
                ComponentPhaseResult(
                    name=comp.name,
                    index=idx,
                    coeffs_total=coeffs_total,
                    coeffs_without=coeffs_without,
                    delta_coeffs=delta_coeffs,
                )
            )

        return results

    @property
    def summary(self) -> None:
        """Prints a summary of the network analysis."""

        def _fmt_db(x: float) -> str:
            if np.isinf(x):
                return "inf"
            return f"{x:7.3f}"

        def _fmt_freq(f: int) -> str:
            if f >= 1e9:
                return f"{f/1e9} GHz"
            if f >= 1e6:
                return f"{f/1e6} MHz"
            return f"{f:.0f} Hz"

        results = self._component_effective_insertion_loss_db()
        freqs = self.network.freqs

        print("\nNETWORK INSERTION LOSS (IL) SUMMARY")
        print("=" * 80)

        # -------------------------------------------------
        # Total insertion loss vs frequency
        # -------------------------------------------------
        print("\nTotal Network IL")
        print("-" * 80)
        print(f"{'Frequency':>12} | {'IL (dB)':>10}")
        print("-" * 80)
        for f, il in zip(freqs, results[0].il_total_db):
            print(f"{_fmt_freq(f):>12} | {_fmt_db(il):>10}")

        # -------------------------------------------------
        # Per component effective contribution (table)
        # -------------------------------------------------
        print("\nEffective IL Contribution per Component")
        print("-" * 80)

        header = f"{'Component':>15} | {'Index':>5}"
        for f in freqs:
            header += f" | {_fmt_freq(f):>10}"
        print(header)
        print("-" * len(header))

        for r in results:
            row = f"{r.name:>15} | {r.index:5d}"
            for v in r.delta_il_db:
                row += f" | {_fmt_db(v):>10}"
            print(row)

        # -------------------------------------------------
        # Aggregate dominance metrics
        # -------------------------------------------------
        print("\nAggregate Effective IL Metrics")
        print("-" * 80)
        print(
            f"{'Component':>15} | {'Mean (dB)':>10} | "
            f"{'RMS (dB)':>10} | {'Max (dB)':>10}"
        )
        print("-" * 80)

        for r in results:
            mean = float(np.mean(r.delta_il_db))
            rms = float(np.sqrt(np.mean(r.delta_il_db**2)))
            maxv = float(np.max(r.delta_il_db))
            print(
                f"{r.name:>15} | "
                f"{_fmt_db(mean):>10} | "
                f"{_fmt_db(rms):>10} | "
                f"{_fmt_db(maxv):>10}"
            )

        # -------------------------------------------------
        # Ranking
        # -------------------------------------------------
        # Ranking (Mean)
        print("\nDominance Ranking by Mean Effective IL")
        print("-" * 80)
        ranking = self._rank_components_by_effective_il(agg="mean")
        for i, (name, score) in enumerate(ranking, start=1):
            print(f"{i:2d}. {name:<15} {_fmt_db(score)} dB")

        # Ranking (RMS)
        print("\nDominance Ranking by RMS Effective IL")
        print("-" * 80)
        ranking = self._rank_components_by_effective_il(agg="rms")
        for i, (name, score) in enumerate(ranking, start=1):
            print(f"{i:2d}. {name:<15} {_fmt_db(score)} dB")

        # Ranking (Max)
        print("\nDominance Ranking by Max Effective IL")
        print("-" * 80)
        ranking = self._rank_components_by_effective_il(agg="max")
        for i, (name, score) in enumerate(ranking, start=1):
            print(f"{i:2d}. {name:<15} {_fmt_db(score)} dB")

        # -------------------------------------------------
        # Phase Error Analysis
        # -------------------------------------------------
        print("\n\nPHASE ERROR ANALYSIS")
        print("=" * 80)

        # Get phase error contributions
        phase_results = self._component_phase_error_contribution()

        # Total network coefficients (same for all components)
        coeffs_total = phase_results[0].coeffs_total if phase_results else None

        if coeffs_total is not None:
            print("\nTotal Network Phase Coefficients (normalized frequency)")
            print("-" * 80)
            print(f"{'Order':>10} | {'Coefficient':>15} | {'Description':>20}")
            print("-" * 80)
            print(f"{'Cubic':>10} | {coeffs_total[0]:>15.6e} | {'Higher-order':>20}")
            print(f"{'Quadratic':>10} | {coeffs_total[1]:>15.6e} | {'Dispersion':>20}")
            print(f"{'Linear':>10} | {coeffs_total[2]:>15.6e} | {'Group Delay':>20}")
            print(
                f"{'Constant':>10} | {coeffs_total[3]:>15.6e} | {'DC Phase Offset':>20}"
            )

            # Per-component contributions
            print("\nPer-Component Phase Error Contribution")
            print("-" * 80)
            print(
                f"{'Component':>15} | {'Cubic (Δc3)':>15} | {'Quadratic (Δc2)':>15} | "
                f"{'Linear (Δc1)':>15} | {'Constant (Δc0)':>15}"
            )
            print("-" * 80)

            for r in phase_results:
                print(
                    f"{r.name:>15} | "
                    f"{r.delta_coeffs[0]:>15.6e} | "
                    f"{r.delta_coeffs[1]:>15.6e} | "
                    f"{r.delta_coeffs[2]:>15.6e} | "
                    f"{r.delta_coeffs[3]:>15.6e}"
                )

            # Dominant contributors by each phase order
            print("\nDominant Phase Error Contributors (by absolute magnitude)")
            print("-" * 80)

            for order_idx, order_name in enumerate(
                ["Cubic", "Quadratic", "Linear", "Constant"]
            ):
                print(f"\n{order_name} Phase Error:")
                ranked = sorted(
                    [(r.name, abs(r.delta_coeffs[order_idx])) for r in phase_results],
                    key=lambda x: x[1],
                    reverse=True,
                )
                for i, (name, mag) in enumerate(ranked[:5], start=1):  # Top 5
                    print(f"  {i}. {name:<15} {mag:>15.6e}")

        print("=" * 80)
