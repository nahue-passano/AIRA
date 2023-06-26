"""Functionality for detecting reflections in a room impulse response (RIR)."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union

import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt


# pylint: disable=too-few-public-methods
class ReflectionDetectionStrategy(ABC):
    """Base interface for a reflection detection algorithm."""

    @staticmethod
    @abstractmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        """Abstract method to be overwritten by concrete implementations of
        reflection detection."""


# pylint: disable=too-few-public-methods
class CorrelationReflectionDetectionStrategy(ReflectionDetectionStrategy):
    """Algorithm for detecting reflections based on the correlation."""

    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implemented method is Scipy's find_peaks")


# pylint: disable=too-few-public-methods
class ThresholdReflectionDetectionStrategy(ReflectionDetectionStrategy):
    """Algorithm for detecting reflections based on a threshold."""

    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implemented method is Scipy's find_peaks")


# pylint: disable=too-few-public-methods
class NeighborReflectionDetectionStrategy(ReflectionDetectionStrategy):
    """Algorithm for detecting reflections based on the surrounding values
    of local maxima."""

    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        """Find local maxima in the intensity magnitude signal.

        Args:
            intensity_magnitude (np.ndarray): intensity magnitude signal.

        Returns:
            np.ndarray: an array with the indeces of the peaks.
        """
        # Drop peak properties ([0]) and direct sound peak ([1])
        return find_peaks(intensity_magnitude)[0]


class WaveletReflectionDetectionStrategy(ReflectionDetectionStrategy):
    """Algorithm for detecting reflections based on the wavelet transform"""

    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        """Find local maxima in the intensity magnitude signal.

        Args:
            intensity_magnitude (np.ndarray): intensity magnitude signal.

        Returns:
            np.ndarray: an array with the indeces of the peaks.
        """
        # Drop peak properties ([0]) and direct sound peak ([1])
        return find_peaks_cwt(intensity_magnitude, widths=np.arange(5, 15))[0]


class ReflectionDetectionStrategies(Enum):
    """Enum class for accessing the existing `ReflectionDetectionStrategy`s"""

    CORRELATION = CorrelationReflectionDetectionStrategy
    THRESHOLD = ThresholdReflectionDetectionStrategy
    SCIPY = NeighborReflectionDetectionStrategy
    WAVELET = WaveletReflectionDetectionStrategy


def detect_reflections(
    intensity: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    detection_strategy: Union[
        ReflectionDetectionStrategy, ReflectionDetectionStrategies
    ] = NeighborReflectionDetectionStrategy,
) -> Tuple[np.ndarray]:
    """Analyze the normalized intensity, azimuth and elevation arrays to look for
    reflections. Timeframes which don't contain a reflection are masked.

    Args:
        intensity (np.ndarray): normalized intensity array.
        azimuth (np.ndarray): array with horizontal angles with respect to the XZ plane.
        elevation (np.ndarray): array with vertical angles with respect to the XY plane.

    Returns:
        intensity (np.ndarray): masked intensities with only reflections different than 0.
        azimuth (np.ndarray): masked intensities with only reflections different than 0.
        elevation (np.ndarray): masked intensities with only reflections different than 0.
    """
    if isinstance(detection_strategy, ReflectionDetectionStrategies):
        detection_strategy = detection_strategy.value

    reflections_indeces = detection_strategy.get_indeces_of_reflections(intensity)
    # Add direct sound
    reflections_indeces = np.insert(reflections_indeces, 0, 0)
    return (
        intensity[reflections_indeces],
        azimuth[reflections_indeces],
        elevation[reflections_indeces],
        reflections_indeces,
    )
