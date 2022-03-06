import numpy as np

class ClockKF(object):
    """
    Object to handle the Kalman Filter for the relative clock states between two DECAR/MRASL 
    UWB modules.

    PARAMETERS:
    -----------
    data: np.array
        Timestamp and ground truth data.
    P0_delta: int
        Initial uncertainty for the clock offset.
    P0_gamma: int
        Initial uncertainty for the clock skew.
    Q_delta: int
        Process variance for the clock offset.
    Q_gamma: int
        Process variance for the clock skew.
    R: int
        Measurement variance.
    """

    _c = 299702547 # speed of light

    def __init__(self, meas, P0_delta=1000000, P0_gamma=1000000,
                 Q_delta=100, Q_gamma=100, R=100):
        """
        Constructor
        """
        