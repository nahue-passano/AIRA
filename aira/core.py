"""Core processing for AIRA module."""
import logging
import time
from dataclasses import dataclass


from aira.engine.input import InputProcessorChain, InputMode
from aira.engine.intensity import convert_bformat_to_intensity
from aira.engine.plot import hedgehog
from aira.engine.reflections import get_hedgehog_arrays
from aira.utils import read_signals_dict


logging.basicConfig(
    filename="aira.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

INTEGRATION_TIME = 0.001
INTENSITY_THRESHOLD = 60


@dataclass
class AmbisonicsImpulseResponseAnalyzer:
    """Main class for analyzing Ambisonics impulse responses"""

    integration_time: float = INTEGRATION_TIME
    intensity_threshold: float = INTENSITY_THRESHOLD
    bformat_frequency_correction: bool = True
    input_builder = InputProcessorChain()

    def _log_settings(self) -> None:
        logging.info("Settings:")
        logging.info(f">> Integration time: {self.integration_time}")
        logging.info(f">> Intensity theshold: {self.intensity_threshold}")

    @staticmethod
    def _timer(time_reference: float) -> float:
        """Counters the running time given a time reference

        Parameters
        ----------
        time_reference : float
            Time reference to be based

        Returns
        -------
        float
            Time difference with time_reference
        """
        return round(time.time() - time_reference, 3)

    def analyze(self, input_dict: dict):
        """Analyzes a set of measurements in Ambisonics format and plots a hedgehog
        with the estimated reflections direction.

        Parameters
        ----------
        input_dict : dict
            Dictionary with all the data needed to analyze a set of measurements
            (paths of the measurements, input mode, channels per file, etc.)
        """
        self._log_settings()
        logging.info("Analyzing input files:")
        for key, value in input_dict.items():
            logging.info(f">> {key}: {value}")

        read_signals_time = time.time()
        signals_dict = read_signals_dict(input_dict)
        logging.info("Run info")
        logging.info(f">> Signals loaded - Done in {self._timer(read_signals_time)} s")

        bformat_preprocessing_time = time.time()
        bformat_signals = self.input_builder.process(input_dict)

        logging.info(
            f">> Input preprocessed - Done in {self._timer(bformat_preprocessing_time)} s"
        )

        bformat_to_intensity_time = time.time()
        intensity, azimuth, elevation = convert_bformat_to_intensity(
            bformat_signals, signals_dict["sample_rate"], self.integration_time
        )

        logging.info(
            f">> Intensity arrays generated - Done in {self._timer(bformat_to_intensity_time)} s"
        )

        hedgehog_arrays_time = time.time()
        masked_intensity, masked_azimuth, masked_elevation = get_hedgehog_arrays(
            intensity, azimuth, elevation
        )

        logging.info(
            f">> Hedgehog arrays generated - Done in {self._timer(hedgehog_arrays_time)} s"
        )

        plot_time = time.time()
        fig = hedgehog(
            masked_intensity,
            masked_azimuth,
            masked_elevation,
            signals_dict["sample_rate"],
            bformat_signals.shape[1] / signals_dict["sample_rate"],
        )

        logging.info(f">> Ploted successfully - Done in {self._timer(plot_time)} s")
        logging.info(f"Run time: {self._timer(read_signals_time)} s")

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
