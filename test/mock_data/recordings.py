from random import randint
import numpy as np


def create_mock_bformat_signal(sample_rate: int, duration_seconds: float) -> np.ndarray:
    return np.random.randint(-32767,
                             32768,
                             size=(4, round(sample_rate*duration_seconds)),
                             dtype=np.int16)
