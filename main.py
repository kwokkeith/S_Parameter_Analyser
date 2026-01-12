from pathlib import Path
from network import Component, Network
import json
from io_utils import load_component

try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path.cwd()

PATH_TO_NETWORK_COMPONENTS = BASE / "data" / "configurations.json"
PATH_TO_S_PARAMETERS_FDR = BASE / "data" / "S_parameters"


def common_freqs_hz(components: list[Component]) -> list[int]:
    if not components:
        return []

    common = set(components[0].S_parameter.keys())
    for comp in components[1:]:
        common &= set(comp.S_parameter.keys())

    return sorted(common)


def main() -> None:
    with open(PATH_TO_NETWORK_COMPONENTS, "r") as f:
        cfg = json.load(f)

    start_hz = int(cfg["global_frequency"]["start_hz"])
    stop_hz = int(cfg["global_frequency"]["stop_hz"])

    components_cfg = cfg["components"]

    # Build components using only the matched frequency rows
    components: list[Component] = []
    for comp_data in components_cfg:
        components.append(
            load_component(comp_data, start_hz, stop_hz, PATH_TO_S_PARAMETERS_FDR)
        )

    global_freqs = common_freqs_hz(components)
    if not global_freqs:
        raise ValueError(
            "No common frequencies across all components within the given start and stop range."
        )

    # Display output network S-parameters
    network = Network(components=components, freqs=global_freqs)
    print("=" * 40)

    # Display network details at common frequencies
    network.display_network(freqs=global_freqs)
    print("=" * 40)

    # Show the exported S-matrix cascaded over the network
    network.export_s_matrix()


if __name__ == "__main__":
    main()
