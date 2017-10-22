from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        affine_relu_out, affine_relu_cache  = affine_relu_forward(X, W1, b1)
        affine_h2_out, affine_h2_cache = affine_forward(affine_relu_out, W2, b2)
        scores = affine_h2_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        softmax_loss_out, d_affine_h2_out = softmax_loss(affine_h2_out, y)
        l2_regularization = lambda LAYER_WEIGHTS: reg * np.sum(LAYER_WEIGHTS ** 2) * 0.5
        reg_loss = l2_regularization(W2) + l2_regularization(W1)
        loss = softmax_loss_out + reg_loss

        d_affine_relu_out, dW2, db2 = affine_backward(d_affine_h2_out, affine_h2_cache)
        dX, dW1, db1 = affine_relu_backward(d_affine_relu_out, affine_relu_cache)

        grads['W2'] = dW2 + reg * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + reg * W1
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        layer_rows = input_dim
        for layer_index in range(self.num_layers):
            layer_cols = hidden_dims[layer_index] if layer_index < len(hidden_dims) else num_classes
            layer_num = '{}'.format(layer_index + 1)
            self.params['W' + layer_num] = weight_scale * np.random.randn(layer_rows, layer_cols)
            self.params['b' + layer_num] = np.zeros(layer_cols)
            if self.use_batchnorm and layer_index < self.num_layers - 1:
                self.params['gamma' + layer_num] = np.ones(layer_cols)
                self.params['beta' + layer_num] = np.zeros(layer_cols)
            layer_rows = layer_cols
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        from functools import partial
        cache = {}
        def memoized(key, backward_func):
            if key not in cache:
                cache.update(backward_func())

            return cache[key]

        layer_mem = {}
        getGradient = lambda grad_key: layer_mem[grad_key]()
        dropout_cache = None
        batchnorm_cache = None
        input_data = X
        AFFINE_ONLY_LAYER = self.num_layers - 1
        for layer_index in range(self.num_layers):
            layer_num = '{}'.format(layer_index + 1)
            Wx_key, bx_key, outx_key = 'W' + layer_num, 'b' + layer_num, 'out' + layer_num
            Wx, bx = self.params[Wx_key], self.params[bx_key]

            if layer_index != AFFINE_ONLY_LAYER:
                input_data, affine_relu_cache = affine_relu_forward(input_data, Wx, bx)
                gammax_key, betax_key = 'gamma' + layer_num, 'beta' + layer_num
                if self.use_batchnorm:
                    gammax, betax = self.params[gammax_key], self.params[betax_key]
                    input_data, batchnorm_cache = batchnorm_forward(input_data, gammax, betax, self.bn_params[layer_index])

                if self.use_dropout:
                    prev_shape = input_data.shape
                    input_data, dropout_cache = dropout_forward(input_data, self.dropout_param)

                def back_hidden(backward_func, forward_cache, cur_dropout_cache, cur_batchnorm_cache):
                    outprev_key = 'out{}'.format(layer_index+2)
                    outx_key = 'out' + layer_num
                    dy = cache[outprev_key]
                    gradient = {}
                    gammax_key, betax_key = 'gamma' + layer_num, 'beta' + layer_num
                    if cur_dropout_cache:
                        dy = dropout_backward(dy, cur_dropout_cache)

                    if cur_batchnorm_cache:
                        dy, dgamma, dbeta = batchnorm_backward(dy, cur_batchnorm_cache)
                        gradient.update({ gammax_key : dgamma, betax_key : dbeta })

                    dOut, dW, db = backward_func(dy, forward_cache)
                    gradient.update({ Wx_key : dW, bx_key : db, outx_key : dOut })

                    return gradient

                compute_gradient = partial(back_hidden, affine_relu_backward, affine_relu_cache, dropout_cache, batchnorm_cache)
                layer_mem[Wx_key] = partial(memoized, Wx_key, compute_gradient)
                layer_mem[bx_key] = partial(memoized, bx_key, compute_gradient)
                layer_mem[gammax_key] = partial(memoized, gammax_key, compute_gradient)
                layer_mem[betax_key] = partial(memoized, betax_key, compute_gradient)
            else:
                input_data, affine_h2_cache = affine_forward(input_data, Wx, bx)
                softmax_loss_out, d_affine_h2_out = softmax_loss(input_data, y)
                def back_out(backward_func, forward_cache):
                    dOut, dW, db = backward_func(d_affine_h2_out, forward_cache)
                    return { Wx_key : dW, bx_key : db, outx_key : dOut }

                layer_mem[Wx_key] = partial(memoized, Wx_key, partial(back_out, affine_backward, affine_h2_cache))
                layer_mem[bx_key] = partial(memoized, bx_key, partial(back_out, affine_backward, affine_h2_cache))

        scores = input_data
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        softmax_loss_out, d_affine_h2_out = softmax_loss(input_data, y)
        l2_regularization = lambda LAYER_WEIGHTS: reg * np.sum(LAYER_WEIGHTS ** 2) * 0.5

        reg_loss = 0

        for layer_index in reversed(range(self.num_layers)):
            layer_num = str(layer_index + 1)
            Wx_key, bx_key = 'W' + layer_num, 'b' + layer_num
            Wx, bx = self.params[Wx_key], self.params[bx_key]
            dWx = getGradient(Wx_key)
            dbx = getGradient(bx_key)
            reg_loss += l2_regularization(Wx)

            grads[Wx_key] = dWx + reg * Wx
            grads[bx_key] = dbx
            if self.use_batchnorm and layer_index < self.num_layers - 1:
                gammax_key, betax_key = 'gamma' + layer_num, 'beta' + layer_num
                grads[gammax_key] = getGradient(gammax_key)
                grads[betax_key] = getGradient(betax_key)

        loss = softmax_loss_out + reg_loss

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
