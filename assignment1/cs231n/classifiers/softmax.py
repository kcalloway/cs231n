import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = y.shape[0]
  for train_index in np.arange(num_train):
    class_scores = X[train_index].dot(W)
    class_scores -= np.max(class_scores)
    probabilities = np.exp(class_scores)/np.sum(np.exp(class_scores))

    for j in xrange(num_classes):
      dW[:, j] +=  probabilities[j] * X[train_index, :]
    dW[:, y[train_index]] -= X[train_index, :]

    loss += -np.log(probabilities[y[train_index]])

  loss /= num_train
  dW /= num_train

  dW += reg * W
  loss += 0.5 * reg * np.sum(W ** 2)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = y.shape[0]

  class_scores = X.dot(W[:])
  class_scores -= np.max(class_scores, axis=1)[:, np.newaxis]
  exp_class_scores = np.exp(class_scores)
  sum_scores = np.sum(exp_class_scores, axis=1)
  probabilities = exp_class_scores/sum_scores[:, np.newaxis]
  correct_probability_indices = y[:, np.newaxis]
  num_classes = class_scores.shape[1]

  loss = np.sum(-np.log(probabilities[range(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)

  dscores = probabilities
  dscores[range(num_train), y] -= 1
  dW = X.T.dot(dscores)
  dW /= num_train
  dW += reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
