import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  sigm = lambda x: 1/(1+np.exp(-x))

  for i in xrange(num_train):
    score = X[i].dot(W)
    score -= np.max(score)
    soft = np.exp(score)/np.sum(np.exp(score))
    loss -= np.log(soft[y[i]])
    
    #print(score, y[i], score.shape, score[5])
    score[y[i]] = 1
    score[np.where(score!=1)] = 0
    score -= soft
    score = -score
    for j in xrange(num_classes):
        dW[:,j] = score[j]*X[i]
 

  # Add regularization to the loss.
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2*reg * W
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
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score -= np.max(score, axis=1, keepdims= True)
  soft = np.exp(score)/np.sum(np.exp(score), axis=1, keepdims=True)
  loss -= np.sum(np.log(soft[xrange(num_train),y]))/num_train
  score[xrange(num_train), y] = 1
  score[np.where(score!=1)] = 0
  score -= soft
  score = -score
  dW = X.T.dot(score)/num_train
    
  loss += reg * np.sum(W * W)
  dW += 2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

