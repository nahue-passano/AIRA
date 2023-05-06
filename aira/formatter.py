from functools import singledispatch
from typing import List

import numpy as np


@singledispatch
def convert_ambisonics_a_to_b(
    front_left_up: np.ndarray,
    front_right_down: np.ndarray,
    back_right_up: np.ndarray,
    back_left_down: np.ndarray,
) -> np.ndarray:
    """Converts Ambisonics A-format to B-format

    Parameters
    ----------
    front_left_up : np.ndarray
            Front Left Up signal from A-format
    front_right_down : np.ndarray
        Front Right Down signal from A-format
    back_right_up : np.ndarray
        Back Right Up signal from A-format
    back_left_down : np.ndarray
        Back Left Down signal from A-format

    Returns
    -------
    np.ndarray
        B-format outputs (W, X, Y, Z)
    """
    
    front = front_left_up + front_right_down
    back = back_left_down + back_right_up
    left = front_left_up + back_left_down
    right = front_right_down + back_right_up
    up = front_left_up + back_right_up
    down = front_right_down + back_left_down

    w_channel = front + back
    x_channel = front - back
    y_channel = left - right
    z_channel = up - down

    return np.array([w_channel, x_channel, y_channel, z_channel])


@convert_ambisonics_a_to_b.register(list)
def _(aformat_channels: List[np.ndarray]) -> np.ndarray:
    """Converts Ambisonics A-format to B-format.

    Parameters
    ----------
    aformat_channels : List[np.ndarray]
        A list containing the 4 channels of A-format Ambisonics in the following order:
            1. Front Left Up
            2. Front Right Down
            3. Back Right Up
            4. Back Left Down

    Returns
    -------
    np.ndarray
        B-format outputs (W, X, Y, Z)
    """
    assert len(aformat_channels) == 4, "Conversion from A-format to B-format requires 4 channels"
    return convert_ambisonics_a_to_b.dispatch(np.ndarray)(
        front_left_up=aformat_channels[0],
        front_right_down=aformat_channels[1],
        back_right_up=aformat_channels[2],
        back_left_down=aformat_channels[3],
    )
