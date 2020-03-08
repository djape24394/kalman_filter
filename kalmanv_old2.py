import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin


# from matplotlib2tikz import save as tikz_save

class KalmanFilter:
    def __init__(self, A: np.ndarray, G: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, x_0=None, P_0=None,
                 save_history=True):
        """
        Kalman filter imlementation, note that x will be 1D ndarray.
            x[k] = Ax[k-1] + Gw[k-1]
            y[k] = Cx[k] + v[k]
        :param A: state transition matrix
        :param G:
        :param C:
        :param Q: process noise covariance matrix(of variable w[k]). If it is scalar, you should pass scalar
        :param R: measurement noise covariance matrix(of variable v[k]). If it is scalar, you should pass scalar
        """
        self.A = A
        self.G = G
        self.C = C
        self.Q = Q
        self.R = R
        self.save_history = save_history

        # list of states with correction X(k|k)
        self.Xcor_list = []
        # current xcor
        self.xcor = None
        # list of prediction states X(k|k-1)
        self.Xpred_list = []
        # current xpred
        self.xpred = None
        # list of measurements y
        self.Y = []
        # Kalman gain K through iterations
        self.K_list = []
        # Kalman gain K
        self.K = None
        self.Ppred_list = []
        self.Pcor_list = []
        # P(k|k-1) = E{(x(k|k-1)-x(k))(x(k|k-1)-x(k))^T}
        self.Ppred = None
        # P(k | k) = E{(x(k | k) - x(k))(x(k | k) - x(k)) ^ T}
        self.Pcor = None
        # E{(y(k) - C*x(k|k-1))(y(k) - C*x(k|k-1)) ^ T}
        self.S = None
        self.S_list = []
        # iteration number
        self.iterarion = 0
        self._initialization(x_0, P_0)

    def _initialization(self, x_0=None, P_0=None):
        if x_0 is None:
            self.xcor = np.zeros(self.A.shape[0])
        else:
            self.xcor = x_0

        if P_0 is None:
            self.Pcor = np.eye(self.A.shape[0])
        else:
            self.Pcor = P_0

        # first element as x(0|0)
        if self.save_history:
            self.Xcor_list = []
            self.Pcor_list = []
            self.Xcor_list.append(self.xcor)
            self.Pcor_list.append(self.Pcor)

    def filter(self, y):
        '''

        :param y: measurement in current iteration, can be float number or 1D ndarray
        :return:
        '''
        # prediction
        self.xpred = self.A @ self.xcor
        self.Ppred = self.A.dot(self.Pcor).dot(self.A.T) + self.G.dot(self.Q).dot(self.G.T)

        # estimation
        self.S = self.C.dot(self.Ppred).dot(self.C.T) + self.R
        if self.S is not np.ndarray:
            self.K = self.Ppred.dot(self.C.T) / self.S
        else:
            self.K = self.Ppred.dot(self.C.T).dot(lin.inv(self.S))
        self.xcor = self.xpred + self.K.dot(y - self.C.dot(self.xpred))
        self.Pcor = (np.eye(self.Ppred.shape[0]) - np.outer(self.K, self.C)).dot(self.Ppred)

        self.iterarion += 1

        if self.save_history:
            # save iteration
            self.Xpred_list.append(self.xpred)
            self.Xcor_list.append(self.xcor)
            self.Ppred_list.append(self.Ppred)
            self.Pcor_list.append(self.Pcor)
            self.S_list.append(self.S)
            self.K_list.append(self.K)

        return self.xpred, self.xcor, self.Ppred, self.Pcor, self.S

    def filter_data(self, data):
        for d in data:
            self.filter(d)
        return self.Xpred_list, self.Xcor_list, self.Ppred_list, self.Pcor_list, self.S_list


if __name__ == '__main__':
    # save_figures_tikz = False
    Ts = 0.1
    n_samples = 100
    # ubrzanje
    a = np.zeros(n_samples)
    a[:30] = 2
    a[30:60] = -3
    # stvarna brzina i pozicija
    v = np.zeros(n_samples)
    p = np.zeros(n_samples)
    for t in np.arange(1, n_samples):
        v[t] = v[t - 1] + Ts * a[t]
        p[t] = p[t - 1] + Ts * v[t] + Ts ** 2 * a[t] / 2

    data = []
    with open('gps_data.txt', 'r') as f:
        for line in f.readlines():
            data.append(float(line))

    t = np.arange(0, 10, 0.1)
    plt.figure()
    # plt.plot(t, a, label='ubrzanje $a$')
    # plt.plot(t, v, label='brzina $v$')
    # plt.plot(t, p, label='pozicija $s$')
    # plt.plot(t, data, label='mjerenje $z$')
    plt.plot(t, a, label='$a$')
    plt.plot(t, v, label='$v$')
    plt.plot(t, p, label='$s$')
    plt.plot(t, data, label='$z$')
    plt.xlabel('vrijeme ($sec$)')
    plt.legend()
    # if save_figures_tikz:
    #     tikz_save('figures\stanjaIMjerenja.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')

    # Kalman
    # x = [p v a]
    A = np.array([[1, Ts, Ts ** 2 / 2], [0, 1, Ts], [0, 0, 1]])
    G = np.array([0, 0, 1]).reshape(3, 1)
    # process noise
    sigma_u = 3
    Q = sigma_u  # np.array([sigma_u])
    # measurement
    C = np.array([1, 0, 0])
    R = 1  # np.array([1])
    x_0 = 0 * np.ones(3)
    P_0 = 10 * np.eye(3)

    kf = KalmanFilter(A, G, C, Q, R, x_0, P_0)

    kf.filter_data(data)

    X = np.array(kf.Xcor_list)
    p_k = X[1:, 0]
    v_k = X[1:, 1]
    a_k = X[1:, 2]  # izbacujem prvi

    K = np.array(kf.K_list)
    Ppred = np.array(kf.Ppred_list)
    Pcor = np.array(kf.Pcor_list[1:])  # izbacujem pocetni

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, a, '--')
    plt.plot(t, a_k)
    plt.ylabel('ubrzanje')
    plt.xlabel('vrijeme ($sec$)')

    plt.subplot(3, 1, 2)
    plt.plot(t, v, '--')
    plt.plot(t, v_k)
    plt.ylabel('brzina')
    plt.xlabel('vrijeme ($sec$)')

    plt.subplot(3, 1, 3)
    plt.plot(t, p, '--')
    plt.plot(t, p_k)
    plt.plot(t, data)
    plt.ylabel('pozicija')
    plt.xlabel('vrijeme ($sec$)')
    # if save_figures_tikz:
    #     tikz_save('figures\estimiranaStanja.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')

    plt.figure()
    # plt.subplot(3, 1, 1)
    plt.plot(t, a, '--')
    plt.plot(t, a_k)
    plt.ylabel('ubrzanje')
    plt.xlabel('vrijeme ($sec$)')
    # if save_figures_tikz:
    #     tikz_save('figures\estimiranaUbrzanje.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')

    plt.figure()
    plt.plot(t, v, '--')
    plt.plot(t, v_k)
    plt.ylabel('brzina')
    plt.xlabel('vrijeme ($sec$)')
    # if save_figures_tikz:
    #     tikz_save('figures\estimiranaBrzina.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')

    plt.figure()
    plt.plot(t, p, '--')
    plt.plot(t, p_k)
    plt.plot(t, data)
    plt.ylabel('pozicija')
    plt.xlabel('vrijeme ($sec$)')
    # if save_figures_tikz:
    #     tikz_save('figures\estimiranaPozicija.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')

    # Kalmanovo pojacanje
    plt.figure()
    plt.plot(t, K[:, 0], label='pozicija')
    plt.plot(t, K[:, 1], label='brzina')
    plt.plot(t, K[:, 2], label='ubrzanje')
    plt.xlabel('vrijeme ($sec$)')
    plt.legend()
    # if save_figures_tikz:
    #     tikz_save('figures\kalmanovoPojacanje.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')
    plt.title('Kalmanovo pojačanje')

    # Varijansa predikcije P(k|k-1) = E{(x(k|k-1)-x(k))(x(k|k-1)-x(k))^T}
    plt.figure()
    plt.plot(t, Ppred[:, 0, 0], label='pozicija')
    plt.plot(t, Ppred[:, 1, 1], label='brzina')
    plt.plot(t, Ppred[:, 2, 2], label='ubrzanje')
    plt.xlabel('vrijeme ($sec$)')
    plt.legend()
    # if save_figures_tikz:
    #     tikz_save('figures\covPredikcije.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')
    plt.title('Varijanse predikcije')

    # Varijansa korekcije P(k | k) = E{(x(k | k) - x(k))(x(k | k) - x(k)) ^ T}
    plt.figure()
    plt.plot(t, Pcor[:, 0, 0], label='pozicija')
    plt.plot(t, Pcor[:, 1, 1], label='brzina')
    plt.plot(t, Pcor[:, 2, 2], label='ubrzanje')
    plt.xlabel('vrijeme ($sec$)')
    plt.legend()
    # if save_figures_tikz:
    #     tikz_save('figures\covKorekcije.tikz',
    #               figureheight='\\figureheight',
    #               figurewidth='\\figurewidth')
    plt.title('Varijanse korekcije')
    plt.show()