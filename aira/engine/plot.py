"""Plotting functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aira.utils.formatter import convert_polar_to_cartesian


def hedgehog(
    reflections_intensity: np.ndarray,
    reflections_azimuth: np.ndarray,
    reflections_elevation: np.ndarray,
    sample_rate: int,
    rir_duration_seconds: float,
):
    """Create a hedgehog plot."""
    time_axis = np.linspace(
        0, rir_duration_seconds / sample_rate, reflections_intensity.shape[0]
    )
    # pylint: disable=invalid-name
    x, y, z = convert_polar_to_cartesian(
        reflections_intensity, reflections_azimuth, reflections_elevation
    )
    plot_df = pd.DataFrame(
        {
            "time": time_axis,
            "intensity": reflections_intensity,
            "azimuth": reflections_azimuth,
            "elevation": reflections_elevation,
            "x": x,
            "y": y,
            "z": z,
        }
    )  # Plotly requires a dataframe input
    plot_df.intensity = plot_df.intensity.astype(float)
    fig = go.Figure(
        data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker={
                "size": 8,
                "color": reflections_intensity,
                "colorscale": "portland",
            },
            line={"color": reflections_intensity, "colorscale": "portland", "width": 6},
        )
    )

    fig.update_layout(template="plotly_dark")
    fig.show()
    return fig
