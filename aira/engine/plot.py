"""Plotting functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aira.utils.formatter import convert_polar_to_cartesian


def hedgehog(
    reflections_time: np.ndarray,
    reflections_intensity: np.ndarray,
    reflections_azimuth: np.ndarray,
    reflections_elevation: np.ndarray,
    sample_rate: int,
    rir_duration_seconds: float,
):
    """Create a hedgehog plot."""
    time_axis = np.arange(0, rir_duration_seconds, 1 / sample_rate)
    # Normalization to first reflection (direct sound)
    reflections_time = time_axis[reflections_time] * 1000
    reflections_time -= reflections_time[0]
    # pylint: disable=invalid-name
    x, y, z = convert_polar_to_cartesian(
        reflections_intensity, reflections_azimuth, reflections_elevation
    )
    reflection_to_direct = 20 * np.log10(reflections_intensity) - 20 * np.log10(
        reflections_intensity.max()
    )
    zero_inserter = lambda i: np.insert(i, np.arange(len(i)), values=0)
    fig = go.Figure(
        data=go.Scatter3d(
            x=zero_inserter(x),
            y=zero_inserter(y),
            z=zero_inserter(z),
            marker={
                "color": zero_inserter(reflections_intensity),
                "colorscale": "portland",
            },
            line={
                "width": 6,
                "color": zero_inserter(reflections_intensity),
                "colorscale": "portland",
            },
            customdata=np.stack(
                (zero_inserter(reflection_to_direct), zero_inserter(reflections_time)),
                axis=-1,
            ),
            hovertemplate="<b>Reflection-to-direct [dB]:</b> %{customdata[0]:.2f} dB <br>"
            + "<b>Time [ms]: </b>%{customdata[1]:.2f} ms <extra></extra>",
        )
    )

    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
    fig.show()
    return fig
