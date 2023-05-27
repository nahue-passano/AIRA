import logging
from typing import List, Union

import numpy as np

from aira.engine.input import (
    AFormatProcessor,
    BFormatProcessor,
    InputProcessorBuilder,
    LSSInputProcessor,
)
from aira.engine.intensity import convert_bformat_to_intensity
from aira.engine.plot import hedgehog
from aira.engine.reflections import get_hedgehog_arrays
from aira.utils import read_signals_dict

logging.basicConfig(
    filename="aira.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s"
)

INTEGRATION_TIME = 0.01
INTENSITY_THRESHOLD = 60


class AmbisonicsImpulseResponseAnalyzer:
    def __init__(
        self,
        integration_time: float,
        intensity_threshold: float,
        bformat_frequency_correction: bool = True,
    ):
        self.integration_time = integration_time
        self.intensity_threshold = intensity_threshold
        self.bformat_frequency_correction = bformat_frequency_correction

    def analyze(self, input_dict: dict):
        # Esto capaz se puede modularizar en otra función/método dentro de input.py
        # Mas que nada para no importar todos los xxxInputProcessor, ya que se basa todo
        # en el dict de entrada
        # TODO: Implementar un logger
        logging.info("[INFO] Analyzing input files:\n")
        for key, value in input_dict.items():
            logging.info(f">> {key}: {value}")

        signals_dict = read_signals_dict(input_dict)

        logging.info("[INFO] Signals loaded successfully")

        input_builder = InputProcessorBuilder()

        if signals_dict["input_mode"] == "lss":
            logging.info("[INFO] Applying LSS processing to get A-Format Signals")
            input_builder.with_processor(
                [LSSInputProcessor(inverse_filter=signals_dict["inverse_filter"])]
            )

        if signals_dict["input_mode"] != "bformat":
            logging.info("[INFO] Applying A-Format processing to get B-Format Signals")
            input_builder.with_processor([AFormatProcessor()])

        if self.bformat_frequency_correction:
            logging.info("[INFO] Applying frequency correction for non-coincident mics")
            input_builder.with_processor(
                [BFormatProcessor(sample_rate=signals_dict["sample_rate"])]
            )

        bformat_signals = input_builder.process(signals_dict["stacked_signals"])

        intensity, azimuth, elevation = convert_bformat_to_intensity(
            bformat_signals, signals_dict["sample_rate"], self.integration_time
        )

        masked_intensity, masked_azimuth, masked_elevation = get_hedgehog_arrays(
            intensity, azimuth, elevation
        )

        fig = hedgehog(
            masked_intensity,
            masked_azimuth,
            masked_elevation,
            signals_dict["sample_rate"],
            bformat_signals.shape[1] / signals_dict["sample_rate"],
        )


if __name__ == "__main__":
    input_dict = {
        "front_left_up": "test/mock_data/soundfield_flu.wav",
        "front_right_down": "test/mock_data/soundfield_frd.wav",
        "back_right_up": "test/mock_data/soundfield_bru.wav",
        "back_left_down": "test/mock_data/soundfield_bld.wav",
        "inverse_filter": "test/mock_data/soundfield_inverse_filter.wav",
        "input_mode": "lss",
        "channels_per_file": 1,
    }

    analyzer = AmbisonicsImpulseResponseAnalyzer(INTEGRATION_TIME, INTENSITY_THRESHOLD)
    analyzer.analyze(input_dict)


"""
input_dict en casos: aformat, lss
input_dict = {
    "front_left_up": <path>,
    "front_right_down": <path>,
    "back_right_up": <path>,
    "back_left_down": <path>,
    "input_mode": "aformat", # options: ["aformat", "bformat" ,"lss"]
    "channels_per_file": 1 # options: [1, 4]
}

input_dict en caso: bformat
input_dict = {
    "w_channel": <path>,
    "x_channel": <path>,
    "y_channel": <path>,
    "z_channel": <path>,
    "input_mode": "bformat",
    "channels_per_file": 4  # options: [1, 4]
}
"""
