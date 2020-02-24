from tensorflow.keras.layers import Conv2D, Dense, Lambda
import tensorflow as tf
import tensorflow.keras.backend as K


class ShakeoutConv2D(Conv2D):

    def __init__(self, tau=0.1, c=0.1, **kwargs):
        assert 0 < tau < 1
        assert 0 < c
        self.tau = tf.constant([tau])
        self.itau = tf.constant([1 / (1 - tau)])
        self.c = c
        super(ShakeoutConv2D, self).__init__(**kwargs)

    def generate_bernoulli_matrix_imatrix(self, shape):
        r_matrix = tf.random.uniform(shape=shape, maxval=1)
        b = tf.math.greater(self.tau, r_matrix)
        ib = tf.math.greater(r_matrix, self.tau)
        f = tf.cast(b, dtype=tf.float32)
        fi = tf.cast(ib, dtype=tf.float32)
        return f, fi

    def call(self, inputs, **kwargs):
        input_dim = inputs.get_shape().as_list()[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        mask, imask = self.generate_bernoulli_matrix_imatrix(kernel_shape)
        mask_sign = K.softsign(self.kernel * mask)
        imask_sign = K.softsign(self.kernel * imask)
        weight = self.c * imask_sign + \
                 self.itau * (self.kernel + self.c * self.tau * mask_sign)
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


class ShakeoutDense(Dense):

    def __init__(self, tau, c, **kwargs):
        assert 0 < tau < 1
        assert 0 < c
        self.tau = tf.constant([tau])
        self.itau = tf.constant([1 / (1 - tau)])
        self.c = c
        super(ShakeoutDense, self).__init__(**kwargs)

    def generate_bernoulli_matrix_imatrix(self, shape):
        r_matrix = tf.random.uniform(shape=shape, maxval=1)
        b = tf.math.greater(self.tau, r_matrix)
        ib = tf.math.greater(r_matrix, self.tau)
        f = tf.cast(b, dtype=tf.float32)
        fi = tf.cast(ib, dtype=tf.float32)
        return f, fi

    def build(self, input_shape):
        super(ShakeoutDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_dim = inputs.get_shape().as_list()[-1]
        kernel_shape = [input_dim, self.units]
        mask, imask = self.generate_bernoulli_matrix_imatrix(kernel_shape)
        mask_sign = K.softsign(self.kernel * mask)
        imask_sign = K.softsign(self.kernel * imask)
        weight = self.c * imask_sign + \
                 self.itau * (self.kernel + self.c * self.tau * mask_sign)
        output = K.dot(inputs, weight)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
