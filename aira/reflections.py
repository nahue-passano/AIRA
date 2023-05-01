from enum import Enum
from abc import ABC, abstractstaticmethod
import numpy as np
from scipy.signal import find_peaks

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
        return find_peaks(intensity_magnitude)[0][1:]  # Drop peak properties ([0]) and direct sound peak ([1])


class ReflectionDetectionStrategies(Enum):
    CORRELATION = CorrelationReflectionDetectionStrategy
    THRESHOLD = ThresholdReflectionDetectionStrategy
    SCIPY = NeighborReflectionDetectionStrategy
