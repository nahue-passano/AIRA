from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union

import numpy as np

from aira.engine.filtering import NonCoincidentMicsCorrection, convolve
from aira.utils import convert_ambisonics_a_to_b


# pylint: disable=too-few-public-methods
class InputStrategy(ABC):
    """Base interface for a reflection detection algorithm."""

    @abstractmethod
    @staticmethod
    def get_ir_array(signals_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Abstract method to be overwritten by concrete implementations of
        input"""


# pylint: disable=too-few-public-methods
class AFormatInputStrategy(InputStrategy):
    """Algorithm to get an array containing the signals in B-Format from
    A-Format"""

    @staticmethod
    def get_ir_array(signals_array: np.ndarray, sample_rate: int) -> np.ndarray:
        bformat_signals = convert_ambisonics_a_to_b(signals_array)
        frequency_corrector = NonCoincidentMicsCorrection(sample_rate)
        bformat_corrected = np.zeros_like(bformat_signals)
        bformat_corrected[0, :] = frequency_corrector.correct_omni(
            bformat_signals[0, :]
        )
        bformat_corrected[1:, :] = frequency_corrector.correct_axis(
            bformat_signals[1:, :]
        )
        return bformat_corrected


# pylint: disable=too-few-public-methods
class BFormatInputStrategy(InputStrategy):
    """Algorithm to get an array containing the signals in B-Format"""

    @staticmethod
    def get_ir_array(signals_array: np.ndarray, sample_rate: int) -> np.ndarray:
        return signals_array


# pylint: disable=too-few-public-methods
class LSSInputStrategy(InputStrategy):
    """Algorithm to get an array containing the signals in B-Format from
    LSS measurements with their inverse filter"""

    @staticmethod
    def get_ir_array(signals_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Available inputs are in A-Format and B-Format")


class InputStrategies(Enum):
    """Enum class for accessing the existing `InputSignalStrategy`s"""

    AFORMAT = AFormatInputStrategy
    BFORMAT = BFormatInputStrategy
    LSS = LSSInputStrategy
