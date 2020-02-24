import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.losses import categorical_crossentropy


def get_loss_with_weight_decay(model: Model, alpha=1e-2):
    vec_weight_list = list()
    for l in model.layers:
        if isinstance(l, Conv2D) or isinstance(l, Dense):
            vec_weight_list.append(tf.reshape(l.trainable_weights[0], [-1]))

    vec_weight = tf.concat(vec_weight_list, axis=0)

    def cross_entropy_with_weight_decay(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred) + alpha * tf.norm(vec_weight)

    return cross_entropy_with_weight_decay
