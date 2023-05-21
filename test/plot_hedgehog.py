import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from aira.formatter import convert_polar_to_cartesian
from aira.intensity import convert_bformat_to_intensity
from aira.plot import hedgehog
from aira.reflections import get_hedgehog_arrays

from mock_data.recordings import bformat_signal_and_samplerate


signals, sample_rate = bformat_signal_and_samplerate()
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
filename = "hedgehog"
if not outputs_directory.exists():
    os.makedirs(outputs_directory)
fig.write_image(str(outputs_directory / filename) + ".webp", format="webp")
fig.write_image(str(outputs_directory / filename) + ".svg", format="svg")
fig.write_html(str(outputs_directory / filename) + ".html", include_plotlyjs=True)
