import numpy as np
from typing import Optional, Iterable
from .component import Component
from .conversions import s_to_abcd, abcd_to_s


class Network:
    _components: list[Component]
    z_0: float = 50.0  # Characteristic impedance in Ohms
    S_matrix: dict[
        np.int64, np.ndarray[np.complex128]
    ]  # Overall S-matrix at different frequencies
    freqs: list[np.int64] = None  # Frequencies at which S-parameters are defined

    def __init__(
        self, components: list[Component], freqs: list[np.int64] = [], z_0: float = 50.0
    ) -> None:
        self._components = components
        self.z_0 = z_0
        self.freqs = freqs
        self._update_Sparameters()

    def _update_Sparameters(self) -> None:
        """Updates the overall S-parameters of the network."""
        self.S_matrix = {}  # Reset S_matrix
        for freq in self.freqs:
            S_11, S_12, S_21, S_22 = self.overall_sparameters(freq=freq)
            self.S_matrix[freq] = np.array([[S_11, S_12], [S_21, S_22]], dtype=complex)

    def insert_component(self, component: Component, idx: Optional[int] = None) -> None:
        """Inserts a component into the network, if idx is given then at the position, otherwise at the end of the network."""
        if idx is not None:
            if idx < 0 or idx > len(self._components):
                raise IndexError(f"Index {idx} is out of range for components list.")
            self._components.insert(idx, component)
        else:
            self._components.append(component)
        self._update_Sparameters()

    def remove_components(self, position_idx: int | Iterable[int]) -> None:
        """Removes a component from the network based on its position index."""
        # Normalize to a list of indices
        if isinstance(position_idx, int):
            indices = [position_idx]
        else:
            indices = list(position_idx)

        for idx in sorted(indices, reverse=True):
            if idx < 0 or idx >= len(self._components):
                raise IndexError(
                    f"Position index {idx} is out of range for components list."
                )
            self._components.pop(idx)
        self._update_Sparameters()

    def export_s_matrix(self) -> dict[np.int64, np.ndarray[np.complex128]]:
        """Exports the overall S-matrix of the network as a 2x2 numpy array."""
        return self.S_matrix

    def display_network(self, freqs: Optional[list[np.int64]] = None) -> None:
        """Outputs the network components and their overall S-parameters."""
        print("Network Components:")

        # Print the network chain
        chain_names = " -> ".join(comp.name for comp in self._components)
        chain_types = " -> ".join(comp.type for comp in self._components)
        print(chain_names)
        print(chain_types)
        print("-" * 40)

        # Print out S-parameters at specified frequencies
        if freqs is not None:
            for freq in freqs:
                if freq not in self.freqs:
                    raise ValueError(
                        f"Frequency at {freq} not within the frequencies of the Network."
                    )
                print(f"Frequency: {freq} Hz")
                for idx, comp in enumerate(self._components):
                    print(f"  Component {idx}: {comp.name}, type = {comp.type} ")
                    print(f"  S-Parameters: \n{comp.S_parameter[freq]}\n")

                # Overall S-Parameters
                overall_S = self.S_matrix[freq]
                print(f" Overall Network S-Parameters at {freq} Hz:")
                print(overall_S)
                print("-" * 40)
        else:
            for idx, comp in enumerate(self._components):
                print(f"  Component {idx}: {comp.name}, type = {comp.type} ")

    def overall_sparameters(
        self, freq: np.int64
    ) -> tuple[complex, complex, complex, complex]:
        """Calculates the overall S-parameters of the network by cascading individual component S-parameters."""
        # Handle empty component list
        if not self._components:
            return 0.0, 0.0, 0.0, 0.0

        total_abcd_matrix = np.array([[1, 0], [0, 1]], dtype=complex)

        # Convert each component S-param to ABCD and cascade
        for comp in self._components:
            comp_S11, comp_S12, comp_S21, comp_S22 = comp.S_parameter[freq].flatten()
            A, B, C, D = s_to_abcd(self.z_0, comp_S11, comp_S12, comp_S21, comp_S22)
            total_abcd_matrix = total_abcd_matrix @ np.array(
                [[A, B], [C, D]], dtype=complex
            )

        A, B, C, D = total_abcd_matrix.flatten()
        return abcd_to_s(self.z_0, A, B, C, D)  # Convert ABCD back to S-parameters
