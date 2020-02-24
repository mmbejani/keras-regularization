from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as k
import tensorflow as tf


class AdaptiveWeightDecay(Callback):

    def __init__(self, model: Model, initial_alpha=1e-10, up_threshold=2.0, down_threshold=1.1, down_rate=0.8,
                 up_rate=2.0, verbose=False):

        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.up_rate_is_done = False
        self.verbose = verbose

        vec_weight_list = list()
        for l in model.layers:
            if isinstance(l, Conv2D) or isinstance(l, Dense):
                vec_weight_list.append(tf.reshape(l.trainable_weights[0], [-1]))
        self.vec_weight = tf.concat(vec_weight_list, axis=0)
        self.alpha = tf.Variable(initial_alpha, dtype=tf.float32)

    def get_loss_function(self):
        def loss_function(y_ture, y_pred):
            return categorical_crossentropy(y_ture, y_pred) + self.alpha * tf.norm(self.vec_weight)

        return loss_function

    def on_epoch_end(self, batch, logs=None):
        val_loss = logs.get('val_loss')
        loss = logs.get('loss')

        v = val_loss / loss

        if v > self.up_threshold:
            alpha_val = k.get_value(self.alpha)
            alpha_val *= self.up_rate
            k.set_value(self.alpha, alpha_val)
            self.up_rate_is_done = True
            if self.verbose:
                print('The alpha value is increased to {0}'.format(alpha_val))

        elif v < self.down_threshold and self.up_rate_is_done:
            alpha_val = k.get_value(self.alpha)
            alpha_val *= self.down_rate
            k.set_value(self.alpha, alpha_val)
            if self.verbose:
                print('The alpha value is decrease to {0}'.format(alpha_val))
