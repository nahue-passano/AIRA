from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

from aira.engine.filtering import NonCoincidentMicsCorrection
from aira.utils import convert_ambisonics_a_to_b


class InputProcessor(ABC):
    @abstractmethod
    def process(self, signals_array: np.ndarray) -> np.ndarray:
        pass


class LSSInputProcessor(InputProcessor):
    def __init__(self, inverse_filter: np.ndarray):
        self.inverse_filter = inverse_filter

    def process(self, signals_array: np.ndarray) -> np.ndarray:
        a_format_signals = np.apply_along_axis(
            lambda array: fftconvolve(array, self.inverse_filter, mode="full"),
            axis=1,
            arr=signals_array,
        )
        return a_format_signals


class AFormatProcessor(InputProcessor):
    def process(self, signals_array: np.ndarray) -> np.ndarray:
        b_format_signals = convert_ambisonics_a_to_b(
            signals_array[0, :],
            signals_array[1, :],
            signals_array[2, :],
            signals_array[3, :],
        )
        return b_format_signals


class BFormatProcessor(InputProcessor):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def process(self, signals_array: np.ndarray) -> np.ndarray:
        frequency_corrector = NonCoincidentMicsCorrection(self.sample_rate)
        b_format_corrected = np.zeros_like(signals_array)
        b_format_corrected[0, :] = frequency_corrector.correct_omni(signals_array[0, :])
        b_format_corrected[1:, :] = frequency_corrector.correct_axis(
            signals_array[1:, :]
        )

        return b_format_corrected


class InputProcessorBuilder:
    def __init__(self):
        self.processors = []

    def with_processor(self, processor: list):
        self.processors.extend(processor)

    def process(self, signals_array: np.ndarray) -> np.ndarray:
        for process_i in self.processors:
            signals_array = process_i.process(signals_array)

        return signals_array
