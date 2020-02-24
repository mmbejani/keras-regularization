from keras.layers import Dense, Conv2D
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import numpy.linalg as linalg
from keras.models import Model
from keras.losses import categorical_crossentropy
import keras.backend as K


class AdaptiveSpectralLoss(Callback):

    def __init__(self, model: Model, val_x: np.ndarray, val_y: np.ndarray, alpha=1e-4):
        self.model = model
        self.alpha = alpha
        self.best_val_loss = 100
        self.val_x = val_x
        self.val_y = val_y
        self.first_epoch_finished = False
        self.w_star = list()
        self.w = list()
        for l in model.layers:
            if isinstance(l, Conv2D) or isinstance(l, Dense):
                self.w_star.append(tf.Variable(l.get_weights()[0]))
                self.w.append(l.trainable_weights[0])

    def get_reg(self):
        w_star_vec = list()
        w_vec = list()
        for i in range(0, len(self.w)):
            w_star_vec.append(tf.reshape(w_star_vec, [-1]))
            w_vec.append(tf.reshape(w_vec, [-1]))
        reg = tf.norm(tf.concat(w_star_vec, axis=0) - tf.concat(w_vec, axis=0))
        return reg

    def get_loss(self, y_true, y_pred):
        reg = self.get_reg()
        return categorical_crossentropy(y_true, y_pred) + self.alpha * reg

    def on_epoch_end(self, batch, logs=None):
        '''if not self.first_epoch_finished:
            return
        indicators = [np.random.randint(0, self.val_x.shape[0] - 1) for _ in range(256)]
        eval = self.model.evaluate(self.val_x[indicators], self.val_y[indicators], batch_size=256, verbose=0)
        val_loss = eval[1]'''
        val_loss = logs.get('val_loss')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print('\n- Find a new suitable weights and updating W* ...')
            for i in range(len(self.w)):
                w_val = K.get_value(self.w[i])
                if len(w_val.shape) > 2:
                    K.set_value(self.w_star[i], self.approximate_svd_tensor(w_val))
                else:
                    K.set_value(self.w_star[i], self.approximation_svd_matrix(w_val))

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

    @staticmethod
    def optimal_d(s):
        variance = np.std(s)
        mean = np.average(s)
        for i in range(s.shape[0] - 1):
            if s[i] < mean + variance:
                return i
        return s.shape[0] - 1
