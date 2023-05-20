from enum import Enum
from abc import ABC, abstractstaticmethod
from typing import Tuple, Union

import numpy as np
from scipy.signal import find_peaks

from aira.intensity import convert_bformat_to_intensity


class ReflectionDetectionStrategy(ABC):
    @abstractstaticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        pass


class CorrelationReflectionDetectionStrategy(ReflectionDetectionStrategy):
    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implemented method is Scipy's find_peaks")


class ThresholdReflectionDetectionStrategy(ReflectionDetectionStrategy):
    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implemented method is Scipy's find_peaks")


class NeighborReflectionDetectionStrategy(ReflectionDetectionStrategy):
    @staticmethod
    def get_indeces_of_reflections(intensity_magnitude: np.ndarray) -> np.ndarray:
        """Find local maxima in the intensity magnitude signal.

        Args:
            intensity_magnitude (np.ndarray): intensity magnitude signal.

        Returns:
            np.ndarray: an array with the indeces of the peaks.
        """
        # Drop peak properties ([0]) and direct sound peak ([1])
        return find_peaks(intensity_magnitude)[0][1:]


class ReflectionDetectionStrategies(Enum):
    CORRELATION = CorrelationReflectionDetectionStrategy
    THRESHOLD = ThresholdReflectionDetectionStrategy
    SCIPY = NeighborReflectionDetectionStrategy


def get_hedgehog_arrays(
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
    mask = np.zeros_like(intensity)
    mask[reflections_indeces] = 1
    return mask * intensity, mask * azimuth, mask * elevation
