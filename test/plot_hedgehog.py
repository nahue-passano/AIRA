import numpy as np
import pandas as pd
import plotly.express as px

from aira.formatter import convert_polar_to_cartesian
from aira.intensity import convert_bformat_to_intensity
from aira.plot import hedgehog
from aira.reflections import get_hedgehog_arrays

from mock_data.recordings import load_mocked_bformat


signals, sample_rate = load_mocked_bformat()
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
    signals.shape[1] / sample_rate
)
fig.write_image("hedge.webp", format="webp")
fig.write_image("hedge.svg", format="svg")
fig.write_html("hedgehog.html", include_plotlyjs=True)
