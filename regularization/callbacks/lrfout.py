from keras.layers import Dense, Input, Conv2D
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from random import randint
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


class Regularization(Callback):
    def __init__(self, model, number_classes, train_input, train_target, batch_size, k=1, patient=3,
                 verbose=True,
                 verbose_condition_number=False):
        y_true = Input(shape=[number_classes])
        self.train_input = train_input
        self.train_target = train_target
        self.error_graph = K.mean(K.pow(y_true - model.output, 2))
        self.error_function = K.function(model.inputs + [y_true], [self.error_graph])
        self.gradient_graph_list = []
        self.gradient_function_list = []
        self.computed_layer_gradient = []
        self.model = model
        self.patient = patient
        self.remain_patient = patient
        self.verbose_condition_number = verbose_condition_number
        self.verbose = verbose
        self.batch_size = batch_size
        self.k = k
        self.monitor = {
            'loss': [],
            'val_loss': [],
            'condition_number': []
        }

        counter = 0
        for l in model.layers:
            if isinstance(l, Dense) or isinstance(l, Conv2D):
                if self.verbose:
                    print('Computing graph of the gradient of layers {0}'.format(counter))
                self.gradient_graph_list.append(K.gradients(self.error_graph, l.trainable_weights[0])[0])
                self.computed_layer_gradient.append(l)
                counter += 1

        self.gradient_functions = K.function(model.inputs + [y_true], self.gradient_graph_list)

    def compute_condition_number(self):
        conditional_number_list = []
        max_cond = 0
        batch_size = 256
        indicator = [randint(0, self.train_target.shape[0] - 1) for _ in range(batch_size)]
        (batch_input, batch_target) = (self.train_input[indicator], self.train_target[indicator])
        gradient_values = self.gradient_functions([batch_input, batch_target])

        for i in range(len(self.computed_layer_gradient)):
            jacobian_matrix = gradient_values[i]
            cond = np.linalg.norm(jacobian_matrix) * np.linalg.norm(
                self.computed_layer_gradient[i].get_weights()[0]) / np.linalg.norm(
                self.error_function([batch_input, batch_target]))
            conditional_number_list.append([cond, self.computed_layer_gradient[i]])
            if cond > max_cond:
                max_cond = cond

        for d in conditional_number_list:
            d[0] /= max_cond
        return conditional_number_list

    def on_epoch_end(self, epoch, logs=None):
        train_error = logs.get('loss')
        validation_error = logs.get('val_loss')

        if validation_error / train_error > 2:
            self.remain_patient -= 1
            if self.verbose:
                print('The algorithm will wait for {0} times'.format(self.remain_patient))
        else:
            self.remain_patient = self.patient

        if self.remain_patient == 0:
            self.remain_patient = self.patient
            counter = 0
            condition_number_list = self.compute_condition_number()
            for kp in condition_number_list:
                if kp[0] > np.random.rand() and kp[0] > 0.7:
                    counter += 1
                    if self.verbose:
                        print('Regularize layers {0} ...'.format(counter))
                    parameters = kp[1].get_weights()
                    l = kp[1]
                    if isinstance(l, Conv2D):
                        w = parameters[0]
                        w = self.approximate_lrf_tensor_kernel_filter_wise(w)
                        if len(parameters) > 1:
                            l.set_weights([w, parameters[1]])
                        else:
                            l.set_weights([w])

                    if isinstance(l, Dense):
                        w = parameters[0]
                        w = self.approximation_nmf_matrix(w)
                        if len(parameters) > 1:
                            l.set_weights([w, parameters[1]])
                        else:
                            l.set_weights([w])

    def on_epoch_begin(self, epoch, logs=None):
        condition_number_list = self.compute_condition_number()
        s = 0
        for cd in condition_number_list:
            s += cd[0]
        self.monitor['condition_number'].append(s)

    def on_train_end(self, logs=None):
        if self.verbose_condition_number:
            plt.plot(self.monitor['condition_number'])
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('SNCN')
            with open('plot-k-{0}.txt'.format(self.k), 'w') as f:
                f.write(str(self.monitor['condition_number']))
            plt.show()

    def approximate_nmf_tensor_filter_wise(self, w):
        w_matrix = np.reshape(w, [w.shape[2], w.shape[3]])
        m = np.min(w_matrix)
        w_matrix -= m
        inner_shape = self.k
        mdl = NMF(n_components=inner_shape, max_iter=10, tol=1.0)
        W = mdl.fit_transform(w_matrix)
        H = mdl.components_
        w_matrix = np.matmul(W, H) + m
        return np.reshape(w_matrix, [1, 1, w_matrix.shape[0], w_matrix.shape[1]])

    def approximate_nmf_tensor_kernel_wise(self, w):
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                m = np.min(w[:, :, i, j])
                w[:, :, i, j] -= m
                mdl = NMF(n_components=self.k, max_iter=10, tol=1.0)
                W = mdl.fit_transform(np.reshape(w[:, :, i, j], [w.shape[0], w.shape[1]]))
                H = mdl.components_
                w[:, :, i, j] = np.matmul(W, H) + m
        return w

    def approximate_lrf_tensor_kernel_filter_wise(self, w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                m = np.min(w[i, j, :, :])
                w[i, j, :, :] -= m
                mdl = NMF(n_components=self.k, max_iter=10, tol=1.0)
                W = mdl.fit_transform(np.reshape(w[i, j, :, :], [w.shape[2], w.shape[3]]))
                H = mdl.components_
                w[i, j, :, :] = np.matmul(W, H) + m
        return w

    def approximation_nmf_matrix(self, w):
        m = np.min(w)
        w -= m
        mdl = NMF(n_components=self.k, max_iter=20, tol=1.0)
        W = mdl.fit_transform(w)
        H = mdl.components_
        return np.matmul(W, H) + m
