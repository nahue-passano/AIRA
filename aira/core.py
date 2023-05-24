from aira.utils import (
    read_aformat,
    convert_polar_to_cartesian,
    convert_ambisonics_a_to_b,
)


class AmbisonicsImpulseResponseAnalyzer:
    def __init__(self, signals_path, integration_time: int, intensity_threshold: int):
        self.signal_paths = signals_path
        self.integration_time = integration_time
        self.intensity_threshold = intensity_threshold

    def calculate(self):
        pass
