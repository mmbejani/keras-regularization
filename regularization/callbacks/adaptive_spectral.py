from keras.layers import Dense, Conv2D
from keras.callbacks import Callback
import numpy as np
import numpy.linalg as linalg
from keras.models import Model


class AdaptiveSpectral(Callback):
    def __init__(self, model: Model, verbose=True):
        self.model = model
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        train_error = logs.get('loss')
        validation_error = logs.get('val_loss')

        if validation_error / train_error > 2:
            counter = 0
            for l in self.model.layers:
                parameters = l.get_weights()
                if self.verbose:
                    print('Regularize layers {0} ...'.format(counter))
                if isinstance(l, Conv2D):
                    w = parameters[0]
                    w = self.approximate_svd_tensor(w)
                    if len(parameters) > 1:
                        l.set_weights([w, parameters[1]])
                    else:
                        l.set_weights([w])

                if isinstance(l, Dense):
                    w = parameters[0]
                    w = self.approximation_svd_matrix(w)
                    if len(parameters) > 1:
                        l.set_weights([w, parameters[1]])
                    else:
                        l.set_weights([w])

    @staticmethod
    def optimal_d(s):
        variance = np.std(s)
        mean = np.average(s)
        for i in range(s.shape[0] - 1):
            if s[i] < mean + variance:
                return i
        return s.shape[0] - 1

    def approximation_svd_matrix(self, w) -> np.ndarray:
        u, s, v = linalg.svd(w)
        d = self.optimal_d(s)
        s = np.diag(s)
        wa = np.dot(u[:, :d], np.dot(s[:d, :d], v[:d, :]))
        return wa

    def approximate_svd_tensor(self, w: np.ndarray) -> np.ndarray:
        w_shape = w.shape
        n1 = w_shape[0]
        n2 = w_shape[1]
        ds = []
        if w_shape[2] == 1 or w_shape[3] == 1:
            return w
        u, s, v = linalg.svd(w)
        for i in range(n1):
            for j in range(n2):
                ds.append(self.optimal_d(s[i, j]))
        d = int(np.mean(ds))
        w = np.matmul(u[..., 0:d], s[..., 0:d, None] * v[..., 0:d, :])
        return w
