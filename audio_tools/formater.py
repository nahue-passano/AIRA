import numpy as np

def A2B_format(FLU: np.ndarray,
               FRD: np.ndarray,
               BRU: np.ndarray,
               BLD: np.ndarray) -> tuple:
    """Converts Ambisonics A-format to B-format

    Parameters
    ----------
    FLU : np.ndarray
            Front Left Up signal from A-format
    FRD : np.ndarray
        Front Right Down signal from A-format
    BRU : np.ndarray
        Back Right Up signal from A-format
    BLD : np.ndarray
        Back Left Down signal from A-format

    Returns
    -------
    tuple
        B-format outputs (W, X, Y, Z)
    """
    
    W = FLU + FRD + BLD + BRU
    X = FLU + FRD - BLD - BRU
    Y = FLU - FRD + BLD - BRU
    Z = FLU - FRD - BLD + BRU

    return W, X, Y, Z
