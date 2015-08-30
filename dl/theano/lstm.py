from collections import OrderedDict
import cPickle as pkl
import sys
import re
import time
import numpy as np
import pandas as pd
from orderedset import OrderedSet

# import relevant theano libraries
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# import internal libraries
# sys.path.insert(0, '../..')
# from util.files import create_path
import pdb

class BinaryClassifierLSTM(object):
    """
    Binary classifier LSTM wrapper using Theano
    """
    def __init__(self):
        # ff: Feed Forward (normal neural net), only useful to put after lstm
        #     before the classifier.
        self.layers = {'lstm': (param_init_lstm, lstm_layer)}

    ###############################################
    ############### INITIALIZATIONS ###############
    ###############################################

    def init_params(options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.

        Output:
        params - dictionary of parameters
        """
        params = OrderedDict()
        # embedding
        randn = np.random.rand(options['n_words'],
                                  options['dim_proj'])
        params['Wemb'] = (0.01 * randn).astype(config.floatX)
        params = get_layer(options['encoder'])[0](options,
                                                  params,
                                                  prefix=options['encoder'])
        # classifier
        params['U'] = 0.01 * np.random.randn(options['dim_proj'],
                                                options['ydim']).astype(config.floatX)
        params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

        return params

    def init_tparams(params):
        """
        Initialize theano parameters

        Output:
        tparams - theano params as dictionary
        """
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams

    def param_init_lstm(options, params, prefix='lstm'):
        """
        Init the LSTM parameter:

        :see: init_params
        """
        W = np.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        params[_p(prefix, 'W')] = W
        U = np.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        params[_p(prefix, 'U')] = U
        b = np.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'b')] = b.astype(config.floatX)

        return params

    ###############################################
    ############## HELPER FUNCTIONS ###############
    ###############################################

    def get_layer(name):
        """
        Return layer as specified in name
        """
        fns = self.layers[name]
        return fns

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

    def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
            preact += x_

            i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
            f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
            o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
            c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                       tparams[_p(prefix, 'b')])

        dim_proj = options['dim_proj']
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[tensor.alloc(np_floatX(0.),
                                                               n_samples,
                                                               dim_proj),
                                                  tensor.alloc(np_floatX(0.),
                                                               n_samples,
                                                               dim_proj)],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps)
        return rval[0]

    def build_model(tparams, options):
        trng = RandomStreams(SEED)

        # Used for dropout.
        use_noise = theano.shared(np_floatX(0.))

        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])
        proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                                prefix=options['encoder'],
                                                mask=mask)
        if options['encoder'] == 'lstm':
            proj = (proj * mask[:, :, None]).sum(axis=0)
            proj = proj / mask.sum(axis=0)[:, None]
        if options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)

        pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

        f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6

        cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

        return use_noise, x, mask, y, f_pred_prob, f_pred, cost


    def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
        """
        If trained model given, this will compute
        the probabilities of new examples.
        """
        n_samples = len(data[0])
        probs = np.zeros((n_samples, 2)).astype(config.floatX)

        n_done = 0

        for _, valid_index in iterator:
            x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                      np.array(data[1])[valid_index],
                                      maxlen=None)
            pred_probs = f_pred_prob(x, mask)
            probs[valid_index, :] = pred_probs

            n_done += len(valid_index)
            if verbose:
                print '%d/%d samples classified' % (n_done, n_samples)

        return probs

    def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
        """
        Just compute the error

        Inputs:
        f_pred - Theano fct computing the prediction
        prepare_data - usual prepare_data for that dataset.
        """
        valid_err = 0
        for _, valid_index in iterator:
            x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                      np.array(data[1])[valid_index],
                                      maxlen=None)
            preds = f_pred(x, mask)
            targets = np.array(data[1])[valid_index]
            valid_err += (preds == targets).sum()
        valid_err = 1. - np_floatX(valid_err) / len(data[0])

        return valid_err

    ###############################################
    ############### DATA PROCESSING ###############
    ###############################################

    def split_sentence(sent):
        """
        Splits sentence into words and punctuations
        """
        return re.findall(r"[\w']+|[.,!?;]", sent)

    def check_dir_files(dataset_path):
        """
        Check if directory exists or files present
        """
        # check if the directory is present
        if not os.path.exists(dataset_path):
            print "The dataset path does not exist"
            sys.exit(1)
        # check if all files are available
        filenames = ['train.txt', 'val.txt', 'test.txt']
        filepaths = []
        for f in filenames:
            filepath = os.path.join(dataset_path,f)
            if not os.path.isfile(filepath):
                print "%s does not exist in the dataset path given" % f
                sys.exit(1)
            filepaths.append(filepath)
        return filepaths

    def build_dict(dataset_path):
        """
        If 'dictionary.pkl' not present,
        Build and return dictionary with all words + indices
        else,
        Return dictionary from 'dictionary.pkl'
        """
        dict_path = os.path.join(dataset_path, 'dictionary.pkl')
        filepaths = check_dir_files(dataset_path)
        if os.path.isfile(dict_path):
            dictionary = pkl.load(dict_path)
            return dictionary
        dictionary = OrderedSet()
        for f in filepaths:
            print "dictionary.pkl file not found - building dictionary..."
            df = pd.read_csv(f)
            sentences = list(df.ix[:,0])
            from s in sentences:
                words = split_sentence(s)
                dictionary = dictionary | OrderedSet(words)
        # write dictionary to pkl file
        pkl.dump(dictionary, open('dictionary.pkl', 'wb'))

        # return dictionary
        return dictionary

    def load_data(dataset_path, n_words=100000, valid_portion=0.1, maxlen=None,
                    sort_by_len=True):
        """
        Loads dataset files from the directory

        Input:
        dataset_path (String) - location of train.txt, val.txt, and test.txt files
        n_words (int) - The number of word to keep in the vocabulary.
            All extra words are set to unknow (1).
        valid_portion (float) - The proportion of the full train set used for
            the validation set.
        maxlen (int or None) - the max sequence length we use in the train/valid set.
        sort_by_len (bool) - Sort by the sequence length for the train,
            valid and test set. This allow faster execution as it cause
            less padding per minibatch. Another mechanism must be used to
            shuffle the train set at each epoch.

        Output:
        (train, valid, test) - each is a tuple w/ list of x-values [0] and y-values [1]
        """
        # check if valid directory + files
        filepaths = check_dir_files(dataset_path)
        # load the dictionary
        dictionary = build_dict(dataset_path)
        # load the data from files
        result = ([],[],[])
        for fidx, f in enumerate(filepaths):
            df = pd.read_csv(f)
            sentences = list(df.ix[:, 0])
            labels = list(df.ix[:, 1])
            from sidx, s in enumerate(sentences):
                words = split_sentence(s)
                # convert to indices
                words = [dictionary.index(w) for w in words]
                sentences[sidx] = words
            result[fidx] = (sentences, labels)
        return result

    ###############################################
    ############### TRAINING & TEST ###############
    ###############################################

    def train_lstm(
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        encoder='lstm',  # TODO: can be removed must be lstm.
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset_path='',

        # Parameter for extra option
        noise_std=0.,
        use_dropout=True,  # if False slightly faster, but worst test error
                           # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
    ):

        # Model options
        model_options = locals().copy()
        print "model options", model_options

        print 'Loading data'
        train, valid, test = load_data(dataset_path=dataset_path, n_words=n_words, valid_portion=0.05,
                                       maxlen=maxlen)
        if test_size > 0:
            # The test set is sorted by size, but we want to keep random
            # size example.  So we must select a random selection of the
            # examples.
            idx = np.arange(len(test[0]))
            np.random.shuffle(idx)
            idx = idx[:test_size]
            test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

        ydim = np.max(train[1]) + 1

        model_options['ydim'] = ydim

        print 'Building model'
        # This create the initial parameters as np ndarrays.
        # Dict name (string) -> np ndarray
        params = init_params(model_options)

        if reload_model:
            load_params('lstm_model.npz', params)

        # This create Theano Shared Variable from the parameters.
        # Dict name (string) -> Theano Tensor Shared Variable
        # params and tparams have different copy of the weights.
        tparams = init_tparams(params)

        # use_noise is for dropout
        (use_noise, x, mask,
         y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

        if decay_c > 0.:
            decay_c = theano.shared(np_floatX(decay_c), name='decay_c')
            weight_decay = 0.
            weight_decay += (tparams['U'] ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        f_cost = theano.function([x, mask, y], cost, name='f_cost')

        grads = tensor.grad(cost, wrt=tparams.values())
        f_grad = theano.function([x, mask, y], grads, name='f_grad')

        lr = tensor.scalar(name='lr')
        f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                            x, mask, y, cost)

        print 'Optimization'

        kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
        kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

        print "%d train examples" % len(train[0])
        print "%d valid examples" % len(valid[0])
        print "%d test examples" % len(test[0])

        history_errs = []
        best_p = None
        bad_count = 0

        if validFreq == -1:
            validFreq = len(train[0]) / batch_size
        if saveFreq == -1:
            saveFreq = len(train[0]) / batch_size

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.clock()
        try:
            for eidx in xrange(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    x = [train[0][t]for t in train_index]

                    # Get the data in np.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = f_grad_shared(x, mask, y)
                    f_update(lrate)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'
                        return 1., 1., 1.

                    if np.mod(uidx, dispFreq) == 0:
                        print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                    if saveto and np.mod(uidx, saveFreq) == 0:
                        print 'Saving...',

                        if best_p is not None:
                            params = best_p
                        else:
                            params = unzip(tparams)
                        np.savez(saveto, history_errs=history_errs, **params)
                        pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                        print 'Done'

                    if np.mod(uidx, validFreq) == 0:
                        use_noise.set_value(0.)
                        train_err = pred_error(f_pred, prepare_data, train, kf)
                        valid_err = pred_error(f_pred, prepare_data, valid,
                                               kf_valid)
                        test_err = pred_error(f_pred, prepare_data, test, kf_test)

                        history_errs.append([valid_err, test_err])

                        if (uidx == 0 or
                            valid_err <= np.array(history_errs)[:,
                                                                   0].min()):

                            best_p = unzip(tparams)
                            bad_counter = 0

                        print ('Train ', train_err, 'Valid ', valid_err,
                               'Test ', test_err)

                        if (len(history_errs) > patience and
                            valid_err >= np.array(history_errs)[:-patience,
                                                                   0].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print 'Early Stop!'
                                estop = True
                                break

                print 'Seen %d samples' % n_samples

                if estop:
                    break

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.clock()
        if best_p is not None:
            zipp(best_p, tparams)
        else:
            best_p = unzip(tparams)

        use_noise.set_value(0.)
        kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
        train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
        valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
        test_err = pred_error(f_pred, prepare_data, test, kf_test)

        print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
        if saveto:
            np.savez(saveto, train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=history_errs, **best_p)
        print 'The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
        print >> sys.stderr, ('Training took %.1fs' %
                              (end_time - start_time))
        return train_err, valid_err, test_err

    def test_lstm(
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        encoder='lstm',  # TODO: can be removed must be lstm.
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset='',
        use_dropout=False,
        # Parameter for extra option
        noise_std=0.
        ):

        # Model options
        model_options = locals().copy()
        print "model options", model_options

        print 'Loading data'
        train, valid, test = load_data(dataset_path=dataset_path, n_words=n_words, valid_portion=0.05,
                                       maxlen=maxlen)
        ydim = np.max(train[1]) + 1
        model_options['ydim'] = ydim

        print 'Building model'
        # This create the initial parameters as np ndarrays.
        # Dict name (string) -> np ndarray
        params = init_params(model_options)
        load_params('lstm_model.npz', params)
        tparams = init_tparams(params)
        # use_noise is for dropout
        (use_noise, x, mask,
         y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

        # get the train, val, and test sets
        kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
        kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
        kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

        print "%d train examples" % len(train[0])
        print "%d valid examples" % len(valid[0])
        print "%d test examples" % len(test[0])

        best_p = unzip(tparams)

        use_noise.set_value(0.)
        pdb.set_trace()
        train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
        valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
        test_err = pred_error(f_pred, prepare_data, test, kf_test)

        print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
        return train_err, valid_err, test_err