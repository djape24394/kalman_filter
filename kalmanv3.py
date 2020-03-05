import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
from tikzplotlib import save as tikz_save

class KalmanFilter:
    def __init__(self, F: np.ndarray, G: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, x_0=None, P_0=None,
                 save_history=True):
        """
        Kalman filter imlementation, note that x will be 1D ndarray.
            x[k] = Fx[k-1] + Gw[k-1]
            y[k] = Hx[k] + v[k]
        :param F: state transition matrix
        :param G:
        :param C:
        :param Q: process noise covariance matrix(of variable w[k]).
        :param R: measurement noise covariance matrix(of variable v[k]).
        """
        pass

if __name__ == "__main__":
    pass