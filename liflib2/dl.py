# -*- coding: utf-8 -*-
'''
Deep learning module of liflib.

Consisting of:

SoftmaxMLP: Softmax MultiLayer Percepton
SoftmaxRNN: Softmax Recurrent Neural Network
RNNLM: A language model based on SoftmaxRNN
'''

import numpy as np
import itertools
import jieba
from nnbase import NNBase
from base import *


class SoftmaxMLP:

    def __init__(self, layers):
        '''
        Initializes a Softmax neural network. 
        '''
        self.n_layers = len(layers)
        self.layer_size = tuple(layers)
        self.w = [None] * (self.n_layers - 1)
        self.b = [None] * (self.n_layers - 1)
        self.fitted = False

    def random_init(self, weight_filler='xavier', bias_filler='constant'):
        for i in xrange(self.n_layers - 1):
            if weight_filler == 'xavier':
                r = math.sqrt(6.0 / (self.layer_size[i]
                                     + self.layer_size[i + 1]))
                self.w[i] = np.random.uniform(-r, r, (self.layer_size[i + 1],
                                                      self.layer_size[i]))
            elif weight_filler == 'constant':
                self.w[i] = np.zeros((self.layer_size[i + 1],
                                      self.layer_size[i]))
            elif weight_filler == 'randn':
                self.w[i] = np.random.randn(self.layer_size[i + 1],
                                            self.layer_size[i])
            else:
                raise ValueError('%s is not a valid weight_filler.' %
                                 weight_filler)

            if bias_filler == 'constant':
                self.b[i] = np.zeros(self.layer_size[i + 1])
            elif bias_filler == 'randn':
                self.b[i] = np.random.randn(self.layer_size[i + 1])
            else:
                raise ValueError('%s is not a valid bias_filler.' %
                                 bias_filler)

    def objective(self, theta, dataset, batch_size=50, randomized=True,
                  regularization=None, _lambda=0):
        '''
        Wrapper used for stochastic gradient descent.
        '''
        self.set_parameters(theta)
        cost = 0
        grad = np.zeros(self.get_parameter_count())
        for i in xrange(batch_size):
            if randomized:
                features, label = dataset.get_random_training_sample()
            else:
                features, label = dataset.get_training_sample()

            target = np.zeros((self.layer_size[-1],))
            target[label] = 1

            sp_cost, grad_w, grad_b = self.feed_forward(features, target)
            lst = []
            for i in xrange(self.n_layers - 1):
                lst.append(grad_w[i].flatten())
                lst.append(grad_b[i].flatten())
            cost += sp_cost / float(batch_size)
            grad += np.concatenate(lst) / float(batch_size)
        if regularization:
            regularization = regularization.lower()
            if regularization == 'l2':
                cost += 0.5 * _lambda * np.sum(theta ** 2)
                grad += _lambda * theta
            else:
                raise ValueError('Unknown regularization: %s' % regularization)
        return cost, grad

    def check_fitted(self):
        '''
        Checks whether the model has been fitted.
        '''
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')

    def forward_propagate(self, features):
        '''
        Calculates the "a" matrices for all layers.
        '''
        a = []
        a.append(features)
        for i in xrange(self.n_layers - 1):
            if i < self.n_layers - 2:
                a.append(np.tanh(self.w[i].dot(a[i]) + self.b[i]))
            else:
                a.append(softmax(self.w[i].dot(a[i]) + self.b[i]))
        return a

    def back_propagate(self, a, target):
        '''
        Calculates the gradients of w and b using BP algorithm.
        '''
        delta = [None] * self.n_layers
        delta[-1] = a[-1] - target
        for i in xrange(self.n_layers - 2, 0, -1):
            delta[i] = (1 - a[i] ** 2) * (self.w[i].T.dot(delta[i + 1]))

        grad_w = [np.outer(delta[i + 1], a[i])
                  for i in xrange(self.n_layers - 1)]
        grad_b = [delta[i + 1] for i in xrange(self.n_layers - 1)]
        return grad_w, grad_b

    def feed_forward(self, features, target):
        a = self.forward_propagate(features)
        cost = -np.log(np.dot(a[-1], target))
        grad_w, grad_b = self.back_propagate(a, target)
        return cost, grad_w, grad_b

    def get_parameter_count(self):
        '''
        Gets the total number of all parameters in this NN.
        '''
        return sum((self.w[i].size + self.b[i].size)
                   for i in xrange(self.n_layers - 1))

    def set_parameters(self, theta):
        '''
        Sets parameters of this NN from argument theta.
        '''
        p = 0
        for i in xrange(self.n_layers - 1):
            self.w[i] = (theta[p:p + self.layer_size[i + 1] * self.layer_size[i]]
                         .reshape(self.layer_size[i + 1], self.layer_size[i]))
            p += self.layer_size[i + 1] * self.layer_size[i]
            self.b[i] = theta[p:p + self.layer_size[i + 1]]
            p += self.layer_size[i + 1]

    def get_parameters(self):
        '''
        Returns a 1D array consist of all parameters in this NN.
        '''
        lst = []
        for i in xrange(self.n_layers - 1):
            lst.append(self.w[i].flatten())
            lst.append(self.b[i].flatten())
        return np.concatenate(lst)

    def fit(self, dataset,
            f_min=sgd,
            init_options={},
            obj_options={},
            f_min_options={}):
        '''
        Fits this NN to given features and labels.
        '''
        self.random_init(**init_options)
        theta = f_min(lambda x:
                      self.objective(x, dataset, **obj_options),
                      self.get_parameters(),
                      **f_min_options)
        self.set_parameters(theta)
        self.fitted = True

    def predict(self, features):
        '''
        Returns the probability of input being in each class given features.
        '''
        self.check_fitted()
        return self.forward_propagate(features)[-1]


class SoftmaxRNN(NNBase):
    """
    Recurrent Neural Network
    Arguments:
        layer_dims : a tuple consisting of size of each layer
        alpha : default learning rate
        rseed : seed for randomization
    """

    def __init__(self, layer_dims, activation="relu",
                 h_filler="identity", w_filler="xavier",
                 alpha=0.005, rseed=10, grad_clipping=True):
        # gradient clipping
        self._grad_clipping = grad_clipping

        # activation function
        if activation not in ACTIVATIONS:
            raise ValueError('Unsupported activation: %s' % activation)
        self._activation = ACTIVATIONS[activation]
        self._derivative = DERIVATIVES[activation]

        # layer dimensions
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims)
        param_dims = dict()
        for i in xrange(1, self.n_layers):
            if i < self.n_layers - 1:
                param_dims['H_%d' % i] = (layer_dims[i], layer_dims[i])
            param_dims['W_%d' % i] = (layer_dims[i], layer_dims[i - 1])
            param_dims['b_%d' % i] = (layer_dims[i],)

        param_dims_sparse = dict()
        NNBase.__init__(self, param_dims, param_dims_sparse,
                        alpha=alpha, rseed=rseed)

        # parameter initialization
        self.H = [None] * self.n_layers
        self.W = [None] * self.n_layers
        self.b = [None] * self.n_layers

        self.dH = [None] * self.n_layers
        self.dW = [None] * self.n_layers
        self.db = [None] * self.n_layers
        for i in xrange(1, self.n_layers):
            if i < self.n_layers - 1:
                self.H[i] = self.params['H_%d' % i]
                self.dH[i] = self.grads['H_%d' % i]
            self.W[i] = self.params['W_%d' % i]
            self.b[i] = self.params['b_%d' % i]
            self.dW[i] = self.grads['W_%d' % i]
            self.db[i] = self.grads['b_%d' % i]

        self._w_filler = WEIGHT_FILLER[w_filler]
        self._h_filler = WEIGHT_FILLER[h_filler]

        for i in xrange(1, self.n_layers):
            self.W[i][:] = self._w_filler(layer_dims[i], layer_dims[i - 1])
        for i in xrange(1, self.n_layers - 1):
            self.H[i][:] = self._h_filler(layer_dims[i], layer_dims[i])

    def _acc_grads(self, xs, ys):
        T = len(xs)
        h = [[np.zeros(d)] for d in self.layer_dims]
        y_hat = [None]
        n = self.n_layers

        for t in xrange(1, T + 1):
            h[0].append(xs[t - 1])
            for l in xrange(1, n - 1):
                h[l].append(self._activation(self.H[l].dot(h[l][t - 1]) +
                                             self.W[l].dot(h[l - 1][t]) + self.b[l]))

            y_hat.append(softmax(self.W[-1].dot(h[-2][t]) + self.b[-1]))

        delta = [np.zeros(d) for d in self.layer_dims]
        gamma = [None for d in self.layer_dims[:-1]]
        for t in xrange(T, 0, -1):
            delta[-1] = y_hat[t].copy()
            delta[-1][ys[t - 1]] -= 1

            self.dW[-1] += np.outer(delta[-1], h[-2][t])
            self.db[-1] += delta[-1].copy()

            for l in xrange(n - 2, 0, -1):
                df = self._derivative(h[l][t])
                delta[l] = df * self.W[l + 1].T.dot(delta[l + 1])
                if t == T:
                    gamma[l] = delta[l].copy()
                elif l == n - 2:
                    gamma[l] = delta[l] + df * self.H[l].T.dot(gamma[l])
                else:
                    gamma[l] = delta[l] + df * (self.H[l].T.dot(gamma[l]) +
                                                self.W[l + 1].T.dot(gamma[l + 1] - delta[l + 1]))

                self.dH[l] += np.outer(gamma[l], h[l][t - 1])
                self.dW[l] += np.outer(gamma[l], h[l - 1][t])
                self.db[l] += gamma[l]

    def _apply_grad_acc(self, alpha=1.0):
        """
        Update parameters with accumulated gradients.

        alpha can be a scalar (as in SGD), or a vector
        of the same length as the full concatenated
        parameter vector (as in e.g. AdaGrad)
        """
        # Dense updates
        if self._grad_clipping:
            self.grads.full *= 5.0 / np.linalg.norm(self.grads.full, ord=2)
        self.params.full -= alpha * self.grads.full

    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.
        """

        J = 0
        h = [np.zeros(d) for d in self.layer_dims]
        for t in xrange(1, len(xs) + 1):
            h[0] = xs[t - 1].copy()
            for l in xrange(1, self.n_layers - 1):
                z = self.H[l].dot(h[l]) + self.W[l].dot(h[l - 1]) + self.b[l]
                h[l] = self._activation(z)

            y_prob = softmax(self.W[-1].dot(h[-2]) + self.b[-1])
            y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
            J += -np.log(max(1e-10, y_prob[ys[t - 1]]))

        return J

    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)
        """
        if not isinstance(X[0][0], np.ndarray):  # single example
            return self.compute_seq_loss(X, Y)
        else:  # multiple examples
            return sum([self.compute_seq_loss(xs, ys)
                        for xs, ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len, Y))
        return J / float(ntot)


class ChineseRNNLM(SoftmaxRNN):

    def __init__(self, filename, hidden_dims=(50,), **kwargs):
        self._sentences = []
        self._word_to_num = {u'<s>': 0, u'</s>': 1}
        with open(filename, 'r') as fp:
            for line in fp:
                raw_sentence = list(jieba.cut(line[:-1]))
                for word in raw_sentence:
                    if word not in self._word_to_num:
                        self._word_to_num[word] = len(self._word_to_num)
                self._sentences.append([0] + list(self._word_to_num[w]
                                                  for w in raw_sentence) + [1])

        self._num_to_word = {self._word_to_num[word]: word
                             for word in self._word_to_num}

        n_words = len(self._word_to_num)
        layer_dims = (n_words,) + hidden_dims + (n_words,)

        super(RNNLM, self).__init__(layer_dims)

    def start_training(self, max_iters=100000,
                       alphaiter=None, printevery=100,
                       costevery=1000, devidx=None):
        X, Y = [], []
        n_words = len(self._word_to_num)
        for st in self._sentences:
            X.append(np.array(list(make_onehot(w, n_words) for w in st[:-1])))
            Y.append(np.array(st[1:]))

        X = np.array(X, dtype=object)
        Y = np.array(Y, dtype=object)

        self.train_sgd(X, Y, idxiter=self.randomiter(max_iters, len(X)),
                       alphaiter=alphaiter, printevery=printevery,
                       costevery=costevery, devidx=devidx)

    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.
        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0
        h = [np.zeros(d) for d in self.layer_dims]
        ys = [init]
        while ys[-1] != end and len(ys) < maxlen:
            h[0] = make_onehot(ys[-1], self.layer_dims[0])
            for l in xrange(1, self.n_layers - 1):
                z = self.H[l].dot(h[l]) + self.W[l].dot(h[l - 1]) + self.b[l]
                h[l] = self._activation(z)

            p = softmax(self.W[-1].dot(h[-2]) + self.b[-1])
            ys.append(multinomial_sample(p))
            J += -np.log(p[ys[-1]])

        return ys, J

    def generate_sentence(self, init=0, end=1):
        st, loss = self.generate_sequence(init, end)
        return ''.join(self._num_to_word[w] for w in st)
