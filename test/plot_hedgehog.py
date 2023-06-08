"""Functional tests related to the generation of the hedgehog plot. Intended
for manual execution."""

import os
from pathlib import Path

from mock_data.recordings import (  # pylint: disable=unused-import
    aformat_signal_and_samplerate,
    bformat_signal_and_samplerate,
)

from aira.engine.intensity import convert_bformat_to_intensity
from aira.engine.plot import hedgehog
from aira.engine.reflections import get_hedgehog_arrays


def test_plot_hedgehog(
    bformat_signal_and_samplerate,
):  # pylint: disable=redefined-outer-name
    """A functional test for generating an example hedgehog based on mocked
    recordings.

    Args:
        bformat_signal_and_samplerate (tuple): a pytest fixture that returns a
        tuple with an array of B format Ambisonics channels and their sample
        rate.
    """
    signals, sample_rate = bformat_signal_and_samplerate
    intensity, azimuth, elevation = convert_bformat_to_intensity(
        signals, sample_rate, 0.01, 4000
    )
    masked_intensity, masked_azimuth, masked_elevation = get_hedgehog_arrays(
        intensity, azimuth, elevation
    )

    fig = hedgehog(
        masked_intensity,
        masked_azimuth,
        masked_elevation,
        sample_rate,
        signals.shape[1] / sample_rate,
    )
    outputs_directory = Path("./outputs")
    output_filename = "hedgehog"
    if not outputs_directory.exists():
        os.makedirs(outputs_directory)
    fig.write_image(str(outputs_directory / output_filename) + ".webp", format="webp")
    fig.write_image(str(outputs_directory / output_filename) + ".svg", format="svg")
    fig.write_html(
        str(outputs_directory / output_filename) + ".html", include_plotlyjs=True
    )
