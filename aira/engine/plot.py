"""Plotting functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aira.utils.formatter import convert_polar_to_cartesian
from aira.engine.intensity import min_max_normalization, intensity_to_dB


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
    normalized_intensities = min_max_normalization(reflections_intensity)
    x, y, z = convert_polar_to_cartesian(
        normalized_intensities, reflections_azimuth, reflections_elevation
    )
    reflection_to_direct = intensity_to_dB(reflections_intensity) - intensity_to_dB(
        reflections_intensity.max()
    )

    zero_inserter = lambda i: np.insert(i, np.arange(len(i)), values=0)
    fig = go.Figure(
        data=go.Scatter3d(
            x=zero_inserter(x),
            y=zero_inserter(y),
            z=zero_inserter(z),
            marker={
                "color": zero_inserter(normalized_intensities),
                "colorscale": "portland",
            },
            line={
                "width": 6,
                "color": zero_inserter(normalized_intensities),
                "colorscale": "portland",
            },
            customdata=np.stack(
                (zero_inserter(reflection_to_direct), zero_inserter(reflections_time)),
                axis=-1,
            ),
            hovertemplate="<b>Reflection-to-direct [dB]:</b> %{customdata[0]:.2f} dB <br>"
            + "<b>Time [ms]: </b>%{customdata[1]:.2f} ms <extra></extra>",
        ),
    )
    camera = {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {"x": 1, "y": 1, "z": 0.7},
    }

    button0 = dict(
        method="relayout",
        args=[{"scene.camera.eye": {"x": 1, "y": 1, "z": 0.7}}],
        label="3D perspective",
    )
    button1 = dict(
        method="relayout",
        args=[{"scene.camera.eye": {"x": 0.0, "y": 0.0, "z": 1.5}}],
        label="X-Y plane",
    )
    button2 = dict(
        method="relayout",
        args=[{"scene.camera.eye": {"x": 0.0, "y": 1.5, "z": 0.0}}],
        label="X-Z plane",
    )
    button3 = dict(
        method="relayout",
        args=[{"scene.camera.eye": {"x": 1.5, "y": 0.0, "z": 0.0}}],
        label="Y-Z plane",
    )
    scene = (
        dict(
            xaxis=dict(
                nticks=4,
                range=[-1, 1],
            ),
            yaxis=dict(
                nticks=4,
                range=[-1, 1],
            ),
            zaxis=dict(
                nticks=4,
                range=[-1, 1],
            ),
        ),
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgb(49,52,56)",
        plot_bgcolor="rgb(49,52,56)",
        scene_camera=camera,
        updatemenus=[dict(buttons=[button0, button1, button2, button3])],
    )
    return fig
