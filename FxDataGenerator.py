# load data using data_loader ~ loader class and directory parametrized
# split data to train/validation/(test?) ~ amount of data in bins parametrized
# normalize data / pct_change
import numpy as np
import  tensorflow.contrib.keras as keras



class FxDataGenerator(object):

    def __init__(self, data, trainDataRatio=0.8, normalizeTool=None, time_samples=24*14, lookahead_period = 4, randSeed=101):
        """
        Initialize FX data generator
        :param data: input data of 3 dimensions [TimeSteps, currencies, OHLC]
        :param trainDataRatio: split to train and test data - ratio of train/all data
        :param normalizeTool: sklearn's tool for data normalization. E.g., sklearn.preprocessing.MinMaxScaler.
        By default, uses None
        Data are not stationary, thus normalize data locally when producing batches
        :param time_samples: number of time steps in learning input data
        :param lookahead_period: specifies the look-ahead amount for data labeling
        :param randSeed numpy random seed
        """
        self.data = data
        self.trainRatio = trainDataRatio
        self.CLOSE_IDX = 3
        self.time_samples = time_samples
        self.lookahead_period = lookahead_period

        samples = data.shape[0]
        self.trainSplitIdx = int(samples*trainDataRatio)

        # shuffle train data
        np.random.seed(randSeed)
        self.train_start_indexes = np.random.permutation(self.trainSplitIdx - self.time_samples)

        self.train_data_idx = 0                        # Current index of train data
        self.test_data_idx = self.trainSplitIdx        # Current index of test data

        # Prepare normalization of each currency - data not stationary, thus need to normalize only locally
        self.normalizeTool = []
        if normalizeTool:
            for i in range(data.shape[1]):
                self.normalizeTool.append(normalizeTool())
                # self.normalizeTool.append(normalizeTool().fit(data[:self.trainSplitIdx, i, :]))

    def generate_rand_train_data(self, batch_size=10, pct_change_threshold=0.001, currency_idx=0, class_mode='binary', normalize=False):
        """
        Generate normalized training data samples with labels of X,y
        :param batch_size: used batch size
        :param pct_change_threshold: threshold to label dataset as 0 - buy / 1 - sell / 2 - noop
        :param currency_idx: currency in question (buy/sell/nothing)
        :param class_mode: 'binary' for binary labels using one-hot-encode, 'multi' for multiple class labels [0,1,2]
        :param normalize: True if data should be normalized using normalization tool defined in constructor
        :return: X - training data matrix of shape [batch_size, time_samples, currencies, OHLC],
        y - vector or matrix of classes, depending on the parameter 'class_mode';
        if class_mode 'binary', return one-hot encoded array [batch, num_classes], else vector of labels
        """
        # get random start point within the training data
        if self.train_data_idx + batch_size > len(self.train_start_indexes):
            self.train_data_idx = 0

        rand_start = self.train_start_indexes[self.train_data_idx:(self.train_data_idx + batch_size)]

        # increment current training index
        self.train_data_idx += batch_size

        # prepare batches of training data
        X = np.zeros([batch_size, self.time_samples] + list(self.data.shape[1:]))
        for i in range(batch_size):
            X[i, :, :, :] = self.data[rand_start[i]:rand_start[i]+self.time_samples, :, :]

        if normalize:
            X = self.normalize_data(X)

        # prepare data labels
        y = self.get_data_labels_N_steps(rand_start,
                                         self.time_samples,
                                         currency_idx,
                                         pct_change_threshold)
        if class_mode == 'binary':
            y = keras.utils.to_categorical(y, num_classes=3)

        return X, y

    def generate_test_data(self, batch_size=10, pct_change_threshold=0.001, currency_idx=0, class_mode='binary', normalize=False):
        """
        Generate normalized testing data samples with labels of X,y
        :param batch_size: used batch size
        :param pct_change_threshold: threshold to label dataset as 0 - buy / 1 - sell / 2 - noop
        :param currency_idx: currency in question (buy/sell/nothing)
        :param class_mode: 'binary' for binary labels using one-hot-encode, 'multi' for multiple class labels [0,1,2]
        :param normalize: True if data should be normalized using normalization tool defined in constructor
        :return: X - training data matrix of shape [batch_size, time_samples, currencies, OHLC],
        y - vector or matrix of classes, depending on the parameter 'class_mode';
        if class_mode 'binary', return one-hot encoded array [batch, num_classes], else vector of labels
        """
        # update the batch size if too large
        new_batch_size = batch_size
        needed_data_size = self.time_samples + self.lookahead_period + 1
        if self.test_data_idx + batch_size > self.data.shape[0] - needed_data_size:
            new_batch_size = self.data.shape[0] - needed_data_size - self.test_data_idx

        # prepare start indexes of all test batches
        incremental_starts = np.arange(self.test_data_idx, (self.test_data_idx + new_batch_size))

        # increment current testing data index
        if new_batch_size == 0:
            self.test_data_idx = self.trainSplitIdx
        else:
            self.test_data_idx += new_batch_size

        # prepare batches of training data
        X = np.zeros([new_batch_size, self.time_samples] + list(self.data.shape[1:]))
        for i in range(new_batch_size):
            X[i, :, :, :] = self.data[incremental_starts[i]:incremental_starts[i]+self.time_samples, :, :]

        if normalize:
            X = self.normalize_data(X)

        # prepare data labels
        y = self.get_data_labels_N_steps(incremental_starts,
                                         self.time_samples,
                                         currency_idx,
                                         pct_change_threshold)
        if class_mode == 'binary':
            y = keras.utils.to_categorical(y, num_classes=3)

        return X, y

    def get_data_labels_1_step(self, rand_start, time_samples, currency_idx, pct_change_threshold):
        """
        Prepare data labels - if the CLOSE price of given currency in next time-step raises above pct_change_threshold
        and not fall in the following step -> buy (0).
        If the situation is the opposite, i.e. close price fall by at least pct_change_threshold and does not raise in
        the following step -> sell(1).
        Otherwise, do nothing (2).
        :param rand_start: vector of data starting indices
        :param time_samples: number of time steps in training data
        :param currency_idx:
        :param pct_change_threshold: threshold to label dataset as 0 - buy / 1 - sell / 2 - noop
        :return: if class_mode 'binary', return one-hot encoded array [batch, num_classes], else vector of labels
        """
        batch_size = len(rand_start)
        y = np.zeros((batch_size, ), dtype=int)
        for i in range(batch_size):
            # get percent changes near the data sample end and two more
            currency_data = self.data[(rand_start[i] + time_samples - 2) : (rand_start[i] + time_samples + 2), currency_idx, self.CLOSE_IDX]
            cur_pct_change = self.pct_change(currency_data)

            if cur_pct_change[-2] > pct_change_threshold:  # and cur_pct_change[-1] > 0:
                y[i] = 0
            elif cur_pct_change[-2] < -pct_change_threshold:  # and cur_pct_change[-1] < 0:
                y[i] = 1
            else:
                y[i] = 2

        return y

    def get_data_labels_N_steps(self, rand_start, time_samples, currency_idx, pct_change_threshold):
        """
        Prepare data labels - if the CLOSE price of given currency in next %lookahead_period% time-steps raises above pct_change_threshold
        -> buy (0).
        If the situation is the opposite, i.e. close price in next 4 time-step falls by at least pct_change_threshold
        -> sell(1).
        Otherwise, do nothing (2).
        :param rand_start: vector of data starting indices
        :param time_samples: number of time steps in training data
        :param currency_idx:
        :param pct_change_threshold: threshold to label dataset as 0 - buy / 1 - sell / 2 - noop
        :return: if class_mode 'binary', return one-hot encoded array [batch, num_classes], else vector of labels
        """
        batch_size = len(rand_start)
        y = np.zeros((batch_size, ), dtype=int)
        for i in range(batch_size):
            # get percent changes near the data sample end and two more
            currency_data = self.data[(rand_start[i] + time_samples) : (rand_start[i] + time_samples + self.lookahead_period + 1), currency_idx, self.CLOSE_IDX]
            cur_pct_change = self.pct_change_N(currency_data, self.lookahead_period)

            if cur_pct_change[0] > pct_change_threshold:
                y[i] = 0
            elif cur_pct_change[0] < -pct_change_threshold:
                y[i] = 1
            else:
                y[i] = 2

        return y

    def pct_change(self, in_data):
        """
        Count percentage change in 1D data
        :param in_data: data to use
        :return: array of pct change in data
        """
        out = np.diff(in_data) / in_data[:-1]
        return out

    def pct_change_N(self, in_data, period=1):
        """
        Count percentage change in 1D data with look-ahead given by parameter period
        :param in_data: data to use
        :param period: specifies the look-ahead amount
        :return: array of pct change in data

        Examples
        --------
        x = np.array([1, 2, 4, 7, 1])
        gen = FxDataGenerator(x)
        gen.pct_change_N(x, 2)
        -> array([ 3. ,  2.5,  -0.75 ])
        """
        short_data = in_data[:-period]
        out = (np.roll(in_data, -period)[:-period] - short_data) / short_data
        return out


    def normalize_data(self, dataToNorm):
        """
        Normalize input data using the normalizationTool defined in constructor
        :param dataToNorm: data to be normalize, size [batch_size, time_samples, currencies, OHLC]
        :return: normalized data of the same shape as the input
        """

        if self.normalizeTool:
            dataOut = np.zeros(dataToNorm.shape)
            for batch in range(dataToNorm.shape[0]):
                for i in range(dataToNorm.shape[2]):
                    dataOut[batch, :, i, :] = self.normalizeTool[i].fit_transform(dataToNorm[batch, :, i, :])
        else:
            dataOut = dataToNorm

        return dataOut

    def get_classes_bias(self, pct_change_threshold, currency_idx):
        """
        Estimate classes weights to deal with unbalanced data classes
        :param pct_change_threshold: threshold for classes identification
        :param currency_idx: currency index the change of which is observed
        :return: dictionary of classes and their weights
        """
        batch_size = 10
        classes = np.zeros(3, )
        num_iter = int(self.data.shape[0] * self.trainRatio / batch_size) + 1
        for i in range(num_iter):
            x, y = self.generate_rand_train_data(batch_size=batch_size,
                                                 pct_change_threshold=pct_change_threshold,
                                                 currency_idx=currency_idx)
            classes[0] += sum(y[:, 0] == 1)
            classes[1] += sum(y[:, 1] == 1)
            classes[2] += sum(y[:, 2] == 1)

        max_weight = max(classes)
        classes_weight = {0: max_weight / classes[0], 1: max_weight / classes[1], 2: max_weight / classes[2]}
        return classes_weight

    # def one_hot_encode(self, vec, vals=3):
    #     """
    #     Encode classification vector into binary matrix
    #     :param vec: label vector
    #     :param vals: number of classes
    #     :return: one-hot encoded vector
    #     """
    #     n = len(vec)
    #     out = np.zeros((n, vals))
    #     out[range(n), vec.astype(int)] = 1
    #     return out


if __name__ == '__main__':
    a = np.array([1, 2, 4, 7, 1])
    gen = FxDataGenerator(a)
    shifted = gen.pct_change_N(a, 2)
    print(shifted)



