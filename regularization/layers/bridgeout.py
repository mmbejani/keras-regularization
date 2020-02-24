from tensorflow.keras.layers import Conv2D, Dense
import tensorflow as tf
import tensorflow.keras.backend as K


class BridgeoutConv2D(Conv2D):

    def __init__(self, p=0.7, q=0.7, padding='same', **kwargs):
        self.p = p
        self.q = q
        super(BridgeoutConv2D, self).__init__(padding=padding, **kwargs)

    def get_bernoulli_mask(self, shape):
        r = tf.random.uniform(shape=shape, maxval=1)
        b = tf.cast(tf.greater(self.p, r), dtype=tf.float32)
        return b

    def call(self, inputs):
        input_dim = inputs.get_shape().as_list()[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        mask = self.get_bernoulli_mask(kernel_shape)
        noise = tf.pow(tf.abs(self.kernel), self.q / 2) * (mask / self.p - 1)
        weight = self.kernel + noise
        outputs = K.conv2d(
            inputs,
            weight,
            strides=self.strides,
            padding=self.padding)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class BridgeoutDense(Dense):

    def __init__(self, p=0.7, q=0.7, **kwargs):
        self.p = p
        self.q = q
        super(BridgeoutDense, self).__init__(**kwargs)

    def get_bernoulli_mask(self, shape):
        r = tf.random.uniform(shape=shape, maxval=1)
        b = tf.cast(tf.greater(self.p, r), dtype=tf.float32)
        return b

    def call(self, inputs):
        input_dim = inputs.get_shape().as_list()[-1]
        kernel_shape = [input_dim, self.units]
        mask = self.get_bernoulli_mask(kernel_shape)
        noise = tf.pow(tf.abs(self.kernel), self.q / 2) * mask
        weight = self.kernel + noise
        output = K.dot(inputs, weight)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
