"""Input preprocessing module."""
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.signal import fftconvolve

from aira.engine.filtering import NonCoincidentMicsCorrection
from aira.utils import convert_ambisonics_a_to_b


# pylint: disable=too-few-public-methods
class InputMode(Enum):
    """Enum class for accessing the existing `InputMode`s"""

    LSS = "lss"
    AFORMAT = "aformat"
    BFORMAT = "bformat"


# pylint: disable=too-few-public-methods
class InputProcessor(ABC):
    """Base interface for inputs processors"""

    @abstractmethod
    def process(self, input_dict: dict) -> dict:
        """Abstract method to be overwritten by concrete implementations of
        input processing."""


# pylint: disable=too-few-public-methods
class LSSInputProcessor(InputProcessor):
    """Processing when input data is in LSS mode"""

    def process(self, input_dict: dict) -> dict:
        """Gets impulse response arrays from Long Sine Sweep (LSS) measurements. The new
        signals are in A-Format.

        Parameters
        ----------
        input_dict : dict
            Dictionary with LSS measurement arrays

        Returns
        -------
        dict
            input_dict overwritten with A-Format signals
        """
        if input_dict["input_mode"] != InputMode.LSS:
            return input_dict

        input_dict["stacked_signals"] = np.apply_along_axis(
            lambda array: fftconvolve(array, input_dict["inverse_filter"], mode="full"),
            axis=1,
            arr=input_dict["stacked_signals"],
        )
        input_dict["input_mode"] = InputMode.AFORMAT

        return input_dict


# pylint: disable=too-few-public-methods
class AFormatProcessor(InputProcessor):
    """Processing when input data is in mode AFORMAT"""

    def process(self, input_dict: dict) -> dict:
        """Gets B-format arrays from A-format arrays. For more details see
        aira.utils.formatter.convert_ambisonics_a_to_b function.

        Parameters
        ----------
        input_dict : dict
            Dictionary with A-format arrays

        Returns
        -------
        dict
            input_dict overwritten with B-format signals
        """
        if input_dict["input_mode"] != InputMode.AFORMAT:
            return input_dict
        input_dict["stacked_signals"] = convert_ambisonics_a_to_b(
            input_dict["stacked_signals"][0, :],
            input_dict["stacked_signals"][1, :],
            input_dict["stacked_signals"][2, :],
            input_dict["stacked_signals"][3, :],
        )
        input_dict["input_mode"] = InputMode.BFORMAT
        return input_dict


# pylint: disable=too-few-public-methods
class BFormatProcessor(InputProcessor):
    """Processin when input data is in BFORMAT mode."""

    def process(self, input_dict: dict) -> dict:
        """Corrects B-format arrays frequency response for non-coincident microphones.

        Parameters
        ----------
        input_dict : dict
            Dictionary with B-format arrays.

        Returns
        -------
        dict
            input_dict overwritten with B-format frequency corrected arrays.
        """
        if input_dict["input_mode"] != InputMode.BFORMAT and not bool(
            input_dict["frequency_correction"]
        ):
            return input_dict

        frequency_corrector = NonCoincidentMicsCorrection(input_dict["sample_rate"])

        input_dict["stacked_signals"][0, :] = frequency_corrector.correct_omni(
            input_dict["stacked_signals"][0, :]
        )
        input_dict["stacked_signals"][1:, :] = frequency_corrector.correct_axis(
            input_dict["stacked_signals"][1:, :]
        )
        input_dict["input_mode"] = InputMode.BFORMAT
        return input_dict


# pylint: disable=too-few-public-methods
class InputProcessorChain:
    """Chain of input processors"""

    def __init__(self):
        self.processors = [LSSInputProcessor(), AFormatProcessor(), BFormatProcessor()]

    def process(self, input_dict: dict) -> np.ndarray:
        """Applies the chain of processors for the input_mode setted.

        Parameters
        ----------
        input_dict : dict
            Contains arrays and input mode data

        Returns
        -------
        np.ndarray
            Arrays processed stacked in single numpy.ndarray object
        """
        for process_i in self.processors:
            input_dict = process_i.process(input_dict)

        return input_dict["stacked_signals"]
