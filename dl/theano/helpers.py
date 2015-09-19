import numpy as np
import theano
from theano import config
from collections import OrderedDict

###############################################
################ DATA HELPERS #################
###############################################

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Shuffle the dataset and return set of indices for minibatches

    Input:
    n - # of total examples
    minibatch_size - size of batch
    (optional) shuffle - shuffle indices before spliting

    Output:
    minibatches - Lists of tuples w/ (idx, []) of minibatches
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def _p(pp, name):
    """
    Return string representation
    """
    return '%s_%s' % (pp, name)

###############################################
############### INIT HELPERS #################
###############################################

def init_tparams(params):
    """
    Initialize theano parameters - make theano.shared vars from param instances

    Output:
    tparams - theano params as dictionary
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

###############################################
############### MODEL HELPERS #################
###############################################

def ortho_weight(ndim):
    """
    Return matrix orthogonal to the weight matrix.
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def load_params(path, params):
    """
    Load the parameters from a param file into local params var

    Output:
    params - dictionary of parameters w/ loaded values
    """
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params