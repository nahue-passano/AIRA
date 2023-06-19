"""Plotting functions."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aira.utils.formatter import convert_polar_to_cartesian
from aira.engine.intensity import min_max_normalization


def hedgehog(
    time_peaks: np.ndarray,
    reflex_to_direct: np.ndarray,
    azimuth_peaks: np.ndarray,
    elevation_peaks: np.ndarray,
):
    """Create a hedgehog plot."""
    time_peaks *= 1000  # seconds to miliseconds
    # pylint: disable=invalid-name

    normalized_intensities = min_max_normalization(reflex_to_direct)
    # pylint: disable=invalid-name
    x, y, z = convert_polar_to_cartesian(
        normalized_intensities, azimuth_peaks, elevation_peaks
    )

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
                (
                    zero_inserter(reflex_to_direct),
                    zero_inserter(time_peaks),
                    zero_inserter(azimuth_peaks),
                    zero_inserter(90 - elevation_peaks),
                ),
                axis=-1,
            ),
            hovertemplate="<b>Reflection-to-direct [dB]:</b> %{customdata[0]:.2f} dB <br>"
            + "<b>Time [ms]: </b>%{customdata[1]:.2f} ms <br>"
            + "<b>Azimuth [°]: </b>%{customdata[2]:.2f}° <br>"
            + "<b>Elevation [°]: </b>%{customdata[3]:.2f}° <extra></extra>",
        ),
    )

    camera, buttons = get_plotly_scenes()

    fig.update_layout(
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
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
