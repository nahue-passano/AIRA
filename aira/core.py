"""Core processing for AIRA module."""
from dataclasses import dataclass
import numpy as np


from aira.engine.input import InputProcessorChain, InputMode
from aira.engine.intensity import (
    convert_bformat_to_intensity,
    analysis_crop,
    integrate_intensity_directions,
    get_intensity_polar_data,
    intensity_thresholding,
)
from aira.engine.plot import hedgehog
from aira.engine.reflections import detect_reflections
from aira.utils import read_signals_dict


INTEGRATION_TIME = 0.001
INTENSITY_THRESHOLD = -60
ANALYSIS_LENGTH = 0.100


@dataclass
class AmbisonicsImpulseResponseAnalyzer:
    """Main class for analyzing Ambisonics impulse responses"""

    integration_time: float = INTEGRATION_TIME
    intensity_threshold: float = INTENSITY_THRESHOLD
    analysis_length: float = ANALYSIS_LENGTH
    bformat_frequency_correction: bool = True
    input_builder = InputProcessorChain()

    def analyze(
        self,
        input_dict: dict,
        integration_time: float = INTEGRATION_TIME,
        intensity_threshold: float = INTENSITY_THRESHOLD,
        analysis_length: float = ANALYSIS_LENGTH,
        show: bool = False,
    ):
        """Analyzes a set of measurements in Ambisonics format and plots a hedgehog
        with the estimated reflections direction.

        Parameters
        ----------
        input_dict : dict
            Dictionary with all the data needed to analyze a set of measurements
            (paths of the measurements, input mode, channels per file, etc.)
        integration_time : float
        intensity_threshold : float
        analysis_length : float
        """

        signals_dict = read_signals_dict(input_dict)
        sample_rate = signals_dict["sample_rate"]

        bformat_signals = self.input_builder.process(input_dict)

        intensity_directions = convert_bformat_to_intensity(
            bformat_signals, sample_rate
        )

        intensity_directions_cropped = analysis_crop(
            analysis_length, sample_rate, intensity_directions
        )

        intensity_windowed = integrate_intensity_directions(
            intensity_directions_cropped, integration_time, sample_rate
        )

        intensity, azimuth, elevation = get_intensity_polar_data(intensity_windowed)

        (
            intensity_peaks,
            azimuth_peaks,
            elevation_peaks,
            reflections_idx,
        ) = detect_reflections(intensity, azimuth, elevation)

        (
            reflex_to_direct,
            azimuth_peaks,
            elevation_peaks,
            reflections_idx,
        ) = intensity_thresholding(
            intensity_threshold,
            intensity_peaks,
            azimuth_peaks,
            elevation_peaks,
            reflections_idx,
        )

        time = np.arange(0, analysis_length, 1 / sample_rate)[reflections_idx]

        fig = hedgehog(time, reflex_to_direct, azimuth_peaks, elevation_peaks)

        if show:
            fig.show()
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

    # # York auditorium
    # data = {
    #     "stacked_signals": "test/mock_data/york_auditorium/s1r2.wav",
    #     "input_mode": InputMode.BFORMAT,
    #     "channels_per_file": 4,
    #     "frequency_correction": False,
    # }

    analyzer = AmbisonicsImpulseResponseAnalyzer()
    analyzer.analyze(data, show=True)
