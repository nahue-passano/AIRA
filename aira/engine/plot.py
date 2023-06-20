"""Plotting functions."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aira.utils.formatter import spherical_to_cartesian
from aira.engine.intensity import min_max_normalization


def hedgehog(
    fig: go.Figure,
    time_peaks: np.ndarray,
    reflex_to_direct: np.ndarray,
    azimuth_peaks: np.ndarray,
    elevation_peaks: np.ndarray,
) -> go.Figure:
    """Create a hedgehog plot."""
    time_peaks *= 1000  # seconds to miliseconds
    normalized_intensities = min_max_normalization(reflex_to_direct)
    # pylint: disable=invalid-name
    x, y, z = spherical_to_cartesian(
        normalized_intensities, azimuth_peaks, elevation_peaks
    )

    fig.add_trace(
        go.Scatter3d(
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
                (
                    zero_inserter(reflex_to_direct),
                    zero_inserter(time_peaks),
                    zero_inserter(azimuth_peaks),
                    zero_inserter(elevation_peaks),
                ),
                axis=-1,
            ),
            hovertemplate="<b>Reflection-to-direct [dB]:</b> %{customdata[0]:.2f} dB <br>"
            + "<b>Time [ms]: </b>%{customdata[1]:.2f} ms <br>"
            + "<b>Azimuth [°]: </b>%{customdata[2]:.2f}° <br>"
            + "<b>Elevation [°]: </b>%{customdata[3]:.2f}° <extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        scene={
            "aspectmode": "cube",
            "xaxis": {"zerolinecolor": "white", "showbackground": False},
            "xaxis_title": " ◀️ Front - Rear ▶",
            "yaxis": {"zerolinecolor": "white", "showbackground": False},
            "yaxis_title": " ◀️ Right - Left ▶",
            "zaxis": {"zerolinecolor": "white", "showbackground": False},
            "zaxis_title": " ◀️ Up - Down ▶",
        },
    )
    return fig


def w_channel(
    fig: go.Figure, time: np.ndarray, w_channel: np.ndarray, ylim: float
) -> go.Figure:
    """_summary_

    Parameters
    ----------
    fig : go.Figure
        _description_
    """
    fig.add_trace(
        go.Scatter(
            x=time,
            y=w_channel,
            customdata=time,
            hovertemplate="<b>Time [ms]:</b> %{customdata:.2f} ms <extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text="Time [ms]", row=2, col=1)
    fig.update_yaxes(title_text="Relative amplitude [dB]", row=2, col=1)


def setup_plotly_layout() -> go.Figure:
    """_summary_

    Parameters
    ----------
    fig : go.Figure
        _description_

    Returns
    -------
    _type_
        _description_
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.05,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
        subplot_titles=("Hedgehog", "W-channel"),
    )

    camera, buttons = get_plotly_scenes()

    fig.update_layout(
        template="plotly_dark",
        margin={"l": 0, "r": 100, "t": 30, "b": 0},
        paper_bgcolor="rgb(49,52,56)",
        plot_bgcolor="rgb(49,52,56)",
        scene_camera=camera,
        updatemenus=[{"buttons": buttons}],
    )

    return fig


def get_plotly_scenes() -> Tuple[Dict]:
    """_summary_

    Returns
    -------
    Tuple[Dict]
        _description_
    """
    camera = {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {"x": 1, "y": 1, "z": 0.7},
    }
    button0 = {
        "method": "relayout",
        "args": [{"scene.camera.eye": {"x": 1, "y": 1, "z": 0.7}}],
        "label": "3D perspective",
    }

    button1 = {
        "method": "relayout",
        "args": [{"scene.camera.eye": {"x": 0.0, "y": 0.0, "z": 1.5}}],
        "label": "X-Y plane",
    }

    button2 = {
        "method": "relayout",
        "args": [{"scene.camera.eye": {"x": 0.0, "y": 1.5, "z": 0.0}}],
        "label": "X-Z plane",
    }

    button3 = {
        "method": "relayout",
        "args": [{"scene.camera.eye": {"x": 1.5, "y": 0.0, "z": 0.0}}],
        "label": "Y-Z plane",
    }
    buttons = [button0, button1, button2, button3]
    return camera, buttons


def zero_inserter(array: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    array : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    return np.insert(array, np.arange(len(array)), values=0)
