from keras.layers import Conv2D, Dense, Reshape
import tensorflow as tf
import numpy as np


class SpectralDropoutConv2D(Conv2D):

    def __init__(self, batch_size, tau=0.1, p=1e-5, activation='relu', **kwargs):
        assert batch_size is not None
        self.tau = tau
        self.p = p
        self.batch_size = batch_size
        super(SpectralDropoutConv2D, self).__init__(activation=activation, **kwargs)

    def get_greater_mask(self, a):
        r = tf.math.greater(a, self.tau)
        ir = tf.math.greater(self.tau, a)
        r = tf.cast(r, dtype=tf.float32)
        ir = tf.cast(ir, dtype=tf.float32)
        return r, ir

    def get_bernoulli_matrix(self, shape):
        shape[0] = self.batch_size
        r_matrix = tf.random.uniform(shape=shape, maxval=1)
        r = tf.math.greater(r_matrix, self.p)
        r = tf.cast(r, dtype=tf.float32)
        return r

    def call(self, inputs, **kwargs):
        output = super().call(inputs)
        shape = self.compute_output_shape(inputs.get_shape().as_list())
        flat_output = Reshape([np.prod(shape[1:])])(output)
        dct_output = tf.signal.dct(flat_output, norm='ortho')
        r, ir = self.get_greater_mask(dct_output)
        b = self.get_bernoulli_matrix(dct_output.get_shape().as_list())
        output_dct = dct_output * r + dct_output * ir * b
        output = tf.signal.idct(output_dct, norm='ortho')
        output = Reshape(shape[1:])(output)
        return output


class SpectralDropoutDense(Dense):

    def __init__(self, batch_size, tau=0.1, p=0.1e-4, activation='relu', **kwargs):
        assert batch_size is not None
        self.batch_size = batch_size
        self.tau = tau
        self.p = p
        super(SpectralDropoutDense, self).__init__(activation=activation, **kwargs)

    def get_greater_mask(self, a):
        r = tf.math.greater(a, self.tau)
        ir = tf.math.greater(self.tau, a)
        r = tf.cast(r, dtype=tf.float32)
        ir = tf.cast(ir, dtype=tf.float32)
        return r, ir

    def get_bernoulli_matrix(self, shape):
        shape[0] = self.batch_size
        r_matrix = tf.random.uniform(shape=shape, maxval=1)
        r = tf.math.greater(r_matrix, self.p)
        r = tf.cast(r, dtype=tf.float32)
        return r

    def call(self, inputs, **kwargs):
        output = super().call(inputs)
        shape = self.compute_output_shape(inputs.get_shape().as_list())
        flat_output = Reshape([np.prod(shape[1:])])(output)
        dct_output = tf.signal.dct(flat_output, norm='ortho')
        r, ir = self.get_greater_mask(dct_output)
        b = self.get_bernoulli_matrix(dct_output.get_shape().as_list())
        output_dct = dct_output * r + dct_output * ir * b
        output = tf.signal.idct(output_dct, norm='ortho')
        output = Reshape(shape[1:])(output)
        return output
