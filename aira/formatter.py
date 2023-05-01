import numpy as np


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

    w_channel = front_left_up + front_right_down + back_left_down + back_right_up
    x_channel = front_left_up + front_right_down - back_left_down - back_right_up
    y_channel = front_left_up - front_right_down + back_left_down - back_right_up
    z_channel = front_left_up - front_right_down - back_left_down + back_right_up

    return np.array([w_channel, x_channel, y_channel, z_channel])
