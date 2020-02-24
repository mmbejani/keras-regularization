from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout


class AdaptiveDropout(Callback):

    def __init__(self, model: Model, up_threshold=2.0, down_threshold=1.1, up_rate=1.5, down_rate=0.8, verbose=False):
        self.dropout_layer = list()
        self.verbose = verbose
        for l in model.layers:
            if isinstance(l, Dropout):
                self.dropout_layer.append(l)

            self.up_threshold = up_threshold
            self.down_threshold = down_threshold
            self.up_rate = up_rate
            self.down_rate = down_rate
            self.is_down_rate_done = False

    def on_epoch_end(self, epoch, logs=None):
        loss_val = logs.get('val_loss')
        loss = logs.get('loss')

        v = loss_val / loss

        if v > self.up_threshold:
            if self.verbose:
                print('The rate values of dropout layers are decreased because of overfitting')
            for l in self.dropout_layer:
                l.rate *= self.down_rate
                l.rate = min(1.0, max(l.rate, 0))
            self.is_down_rate_done = True

        elif v < self.down_threshold and self.is_down_rate_done:
            if self.verbose:
                print('The rate values of dropout layers are increased because of underfitting')
            for l in self.dropout_layer:
                l.rate *= self.up_rate
                l.rate = min(1.0, max(l.rate, 0))
