"""Core processing for AIRA module."""
from dataclasses import dataclass


from aira.engine.input import InputProcessorChain, InputMode
from aira.engine.intensity import convert_bformat_to_intensity
from aira.engine.plot import hedgehog
from aira.engine.reflections import get_hedgehog_arrays
from aira.utils import read_signals_dict

INTEGRATION_TIME = 0.01
INTENSITY_THRESHOLD = 60


@dataclass
class AmbisonicsImpulseResponseAnalyzer:
    """Main class for analyzing Ambisonics impulse responses"""

    integration_time: float = INTEGRATION_TIME
    intensity_threshold: float = INTENSITY_THRESHOLD
    bformat_frequency_correction: bool = True
    input_builder = InputProcessorChain()

    def analyze(self, input_dict: dict):
        """Analyzes a set of measurements in Ambisonics format and plots a hedgehog
        with the estimated reflections direction.

        Parameters
        ----------
        input_dict : dict
            Dictionary with all the data needed to analyze a set of measurements
            (paths of the measurements, input mode, channels per file, etc.)
        """
        print("Analyzing input files:")
        for key, value in input_dict.items():
            print(f">> {key}: {value}")

        signals_dict = read_signals_dict(input_dict)
        print("Run info")
        print(">> Signals loaded")

        bformat_signals = self.input_builder.process(input_dict)

        print(">> Input preprocessed")

        intensity, azimuth, elevation = convert_bformat_to_intensity(
            bformat_signals, signals_dict["sample_rate"], self.integration_time
        )

        print(">> Intensity arrays generated")

        (
            masked_intensity,
            masked_azimuth,
            masked_elevation,
            reflections_indeces,
        ) = get_hedgehog_arrays(intensity, azimuth, elevation)

        print(">> Hedgehog arrays generated")

        fig = hedgehog(
            reflections_indeces,
            masked_intensity,
            masked_azimuth,
            masked_elevation,
            signals_dict["sample_rate"],
            bformat_signals.shape[1] / signals_dict["sample_rate"],
        )

        print(f">> Ploted successfully")

        return fig


if __name__ == "__main__":
    # Regio theater
    data = {
        "front_left_up": "test/mock_data/regio_theater/soundfield_flu.wav",
        "front_right_down": "test/mock_data/regio_theater/soundfield_frd.wav",
        "back_right_up": "test/mock_data/regio_theater/soundfield_bru.wav",
        "back_left_down": "test/mock_data/regio_theater/soundfield_bld.wav",
        "inverse_filter": "test/mock_data/regio_theater/soundfield_inverse_filter.wav",
        "input_mode": InputMode.LSS,
        "channels_per_file": 1,
        "frequency_correction": True,
    }

    # York auditorium
    # data = {
    #     "stacked_signals": "test/mock_data/york_auditorium/s1r2.wav",
    #     "input_mode": InputMode.BFORMAT,
    #     "channels_per_file": 4,
    #     "frequency_correction": True,
    # }

    analyzer = AmbisonicsImpulseResponseAnalyzer()
    analyzer.analyze(data)
