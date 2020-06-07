import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin


class KalmanFilter:
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R,
                 save_history=False, dtype=np.float64):
        """
        Kalman filter imlementation, note that x will be 1D ndarray.
            x[k] = Fx[k-1] + w[k-1]
            y[k] = Hx[k] + v[k]
        Inputs:
        - F: state transition model
        - H: measurement model
        - Q: process noise covariance matrix(of variable w[k]).
        - R: measurement noise covariance matrix(of variable v[k]).
        """
        self.F = dtype(F)
        self.H = dtype(H)
        self.Q = dtype(Q)
        self.R = dtype(R)
        self.dtype = dtype
        self.save_history = save_history

    def prediction(self, x, P_cor):
        """
        Prediction step of the linear Kalman filter
        """
        x_pred = self.F @ x
        P_pred = self.F @ P_cor @ self.F.T + self.Q
        return x_pred, P_pred

    def innovation(self, z, x_pred, P_pred):
        """
        Innovation
        """
        S = self.H @ P_pred @ self.H.T + self.R
        residual = z - self.H @ x_pred
        return residual, S

    def correction(self, x_pred, P_pred, residual, S):
        """
        The correction step of the Kalman filter
        """
        if S is np.ndarray:
            K = P_pred @ self.H.T @ np.linalg.inv(S)
        else:
            K = P_pred @ self.H.T / S
        x_cor = x_pred + K.dot(residual)
        P_corr = (np.eye(x_pred.shape[0]) - np.outer(K, self.H)) @ P_pred
        return x_cor, P_corr, K

    def filter_data(self, Z, x_0=None, P_0=None):
        if x_0 is None:
            x_0 = np.zeros(self.F.shape[0], dtype=self.dtype)

        if P_0 is None:
            P_0 = np.eye(self.F.shape[0], dtype=self.dtype)

        x_corr_list = []
        if self.save_history is True:
            x_pred_history = []
            P_pred_history = []
            residual_history = []
            S_history = []
            P_corr_history = []
            K_history = []
        x_corr = x_0
        P_corr = P_0
        for z in Z:
            x_pred, P_pred = self.prediction(x_corr, P_corr)
            residual, S = self.innovation(z, x_pred, P_pred)
            x_corr, P_corr, K = self.correction(x_pred, P_pred, residual, S)
            x_corr_list.append(x_corr)
            if self.save_history is True:
                x_pred_history.append(x_pred)
                P_pred_history.append(P_pred)
                residual_history.append(residual)
                S_history.append(S)
                P_corr_history.append(P_corr)
                K_history.append(K)
        if self.save_history is False:
            return x_corr_list, None
        else:
            history = (x_pred_history, P_pred_history, residual_history, S_history, P_corr_history, K_history)
            return x_corr_list, history


if __name__ == "__main__":
    # Sampling rate
    Ts = 0.1

    #  Number of steps for which the observation will be generated
    n_samples = 100

    # Generating Acceleration
    a = np.zeros(n_samples, dtype=np.float64)
    a[:30] = 2
    a[30:60] = -3

    # Generating velocity and speed based on acceleration, measurement set used later is created exactly by these
    # equations, and we reconstruct these states because of illustrations to make more clear the behavior of the filter
    v = np.zeros(n_samples, dtype=np.float64)
    v[0] = 0
    s = np.zeros(n_samples, dtype=np.float64)
    s[0] = 0
    for i in range(1, n_samples):
        v[i] = v[i - 1] + Ts * a[i]
        s[i] = s[i - 1] + Ts * v[i] + 0.5 * Ts ** 2 * a[i]

    # Read the data from the input file (data is created by adding additive noise to vector s created earlier)
    data = []
    with open('gps_data.txt', 'r') as f:
        for line in f.readlines():
            data.append(float(line))

    t = np.arange(0, 10, 0.1)

    plt.figure()
    plt.plot(t, a, label='$a$')
    plt.plot(t, v, label='$v$')
    plt.plot(t, s, label='$s$')
    plt.plot(t, data, label='$z$')
    plt.xlabel('time [$sec$]')
    plt.legend()
    plt.grid(True)

    # Transition model, assuming constant acceleration(which actually is constant acceleration)
    F = np.array([[1, Ts, Ts ** 2 / 2], [0, 1, Ts], [0, 0, 1]])
    # Measurement model
    H = np.array([1, 0, 0])
    # Process noise
    sigma_w = 3
    G = np.array([Ts ** 2 / 2, Ts, 1])
    Q = np.outer(G, G) * sigma_w

    # Measurement noise
    R = 1.0

    x_0 = np.zeros(3)
    P_0 = np.eye(3)

    kalman_filter = KalmanFilter(F, H, Q, R, save_history=True)

    x_corr_list, history = kalman_filter.filter_data(data, x_0, P_0)

    # Plotting the state estimation results
    X = np.array(x_corr_list)
    s_k = X[:, 0]
    v_k = X[:, 1]
    a_k = X[:, 2]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, a, '--')
    plt.plot(t, a_k)
    plt.ylabel('acceleration')
    plt.xlabel('time [$sec$]')

    plt.subplot(3, 1, 2)
    plt.plot(t, v, '--')
    plt.plot(t, v_k)
    plt.ylabel('speed')
    plt.xlabel('time [$sec$]')

    plt.subplot(3, 1, 3)
    plt.plot(t, data, label='$z$')
    plt.plot(t, s, '--', label='$s$')
    plt.plot(t, s_k, label='S', color='r')
    plt.ylabel('position')
    plt.xlabel('time [$sec$]')

    # Plotting the statistics and Kalman gain
    x_pred_history, P_pred_history, residual_history, S_history, P_corr_history, K_history = history

    # Plotting the residual with its standard deviations
    resid = np.array(residual_history)
    S = np.array(S_history)
    stds = np.sqrt(S)
    plt.figure()
    plt.plot(t, stds, '--', color='k')
    plt.plot(t, -stds, '--', color='k')
    plt.plot(t, 2 * stds, '--', color='k')
    plt.plot(t, -2 * stds, '--', color='k')
    plt.plot(t, resid)
    plt.ylabel('Innovation and innovation standard deviation bounds')
    plt.xlabel('time [$sec$]')

    # Plotting the Kalman gain
    K = np.array(K_history)
    plt.figure()
    plt.plot(t, K[:, 0], label='s')
    plt.plot(t, K[:, 1], label='v')
    plt.plot(t, K[:, 2], label='a')
    plt.ylabel('')
    plt.xlabel('time [$sec$]')
    plt.title('Kalman gains')
    plt.legend()
    plt.grid(True)

    # Plotting the variances from correction covariance matrix
    P_corr = np.array(P_corr_history)
    plt.figure()
    plt.plot(t, P_corr[:, 0, 0], label='var{s}')
    plt.plot(t, P_corr[:, 1, 1], label='var{v}')
    plt.plot(t, P_corr[:, 2, 2], label='var{a}')
    plt.ylabel('')
    plt.xlabel('time [$sec$]')
    plt.title('Variances of estimations')
    plt.legend()
    plt.grid(True)
    plt.show()

