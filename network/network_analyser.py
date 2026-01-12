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

        print("=" * 80)
