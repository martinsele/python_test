# load data using data_loader ~ loader class and directory parametrized
# split data to train/validation/(test?) ~ amount of data in bins parametrized
# normalize data / pct_change
import numpy as np
import  tensorflow.contrib.keras as keras



class FxDataGenerator(object):

    def __init__(self, data, trainDataRatio=0.8, normalizeTool=None, randSeed=101):
        """
        Initialize FX data generator
        :param data: input data of 3 dimensions [TimeSteps, currencies, OHLC]
        :param trainDataRatio: split to train and test data - ratio of train/all data
        :param normalizeTool: sklearn's tool for data normalization. E.g., sklearn.preprocessing.MinMaxScaler.
        By default, uses None
        Data are not stationary, thus normalize data locally when producing batches
        """
        self.data = data
        self.trainRatio = trainDataRatio
        self.CLOSE_IDX = 3

        np.random.seed(randSeed)

        samples = data.shape[0]
        self.trainSplitIdx = int(samples*trainDataRatio)

        # Prepare normalization of each currency - data not stationary, thus need to normalize only locally
        self.normalizeTool = []
        if normalizeTool:
            for i in range(data.shape[1]):
                self.normalizeTool.append(normalizeTool())
                # self.normalizeTool.append(normalizeTool().fit(data[:self.trainSplitIdx, i, :]))

    def generate_rand_train_data(self, time_samples=24*14, batch_size=10, pct_change_threshold=0.001, currency_idx=0, class_mode='binary', normalize=False):
        """
        Generate normalized training data samples with labels of X,y
        :param time_samples: number of time steps in X data
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
        rand_start = np.random.randint(0, self.trainSplitIdx-time_samples, (batch_size, ))

        # prepare batches of training data
        X = np.zeros([batch_size, time_samples] + list(self.data.shape[1:]))
        for i in range(batch_size):
            X[i, :, :, :] = self.data[rand_start[i]:rand_start[i]+time_samples, :, :]
            if normalize:
                X = self.normalize_data(X)

        # prepare data labels
        y = self.get_data_labels_N_steps(rand_start,
                                         time_samples,
                                         currency_idx,
                                         pct_change_threshold)
        if class_mode == 'binary':
            y = keras.utils.to_categorical(y, num_classes=3)

        return X, y

    def generate_test_data(self, time_samples=24*14, batch_size=10, pct_change_threshold=0.001, currency_idx=0, class_mode='binary', normalize=False):
        """
        Generate normalized testing data samples with labels of X,y
        :param time_samples: number of time steps in X data
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
        rand_start = np.random.randint(0, self.trainSplitIdx-time_samples, (batch_size, ))

        # prepare batches of training data
        X = np.zeros([batch_size, time_samples] + list(self.data.shape[1:]))
        for i in range(batch_size):
            X[i, :, :, :] = self.data[rand_start[i]:rand_start[i]+time_samples, :, :]
            if normalize:
                X = self.normalize_data(X)

        # prepare data labels
        y = self.get_data_labels_N_steps(rand_start,
                                         time_samples,
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
        y = np.zeros((batch_size, ))
        for i in range(batch_size):
            # get percent changes near the data sample end and two more
            currency_data = self.data[(rand_start[i] + time_samples - 2) : (rand_start[i] + time_samples + 2), currency_idx, self.CLOSE_IDX]
            cur_pct_change = self.pct_change(currency_data)

            if cur_pct_change[-2] > pct_change_threshold: # and cur_pct_change[-1] > 0:
                y[i] = 0
            elif cur_pct_change[-2] < -pct_change_threshold: # and cur_pct_change[-1] < 0:
                y[i] = 1
            else:
                y[i] = 2

        return y

    def get_data_labels_N_steps(self, rand_start, time_samples, currency_idx, pct_change_threshold, period=4):
        """
        Prepare data labels - if the CLOSE price of given currency in next %period% time-steps raises above pct_change_threshold
        -> buy (0).
        If the situation is the opposite, i.e. close price in next 4 time-step falls by at least pct_change_threshold
        -> sell(1).
        Otherwise, do nothing (2).
        :param rand_start: vector of data starting indices
        :param time_samples: number of time steps in training data
        :param currency_idx:
        :param pct_change_threshold: threshold to label dataset as 0 - buy / 1 - sell / 2 - noop
        :param period: specifies the look-ahead amount
        :return: if class_mode 'binary', return one-hot encoded array [batch, num_classes], else vector of labels
        """
        batch_size = len(rand_start)
        y = np.zeros((batch_size, ))
        for i in range(batch_size):
            # get percent changes near the data sample end and two more
            currency_data = self.data[(rand_start[i] + time_samples) : (rand_start[i] + time_samples + 2*period), currency_idx, self.CLOSE_IDX]
            cur_pct_change = self.pct_change_N(currency_data, period)

            if cur_pct_change[0] > pct_change_threshold: # and cur_pct_change[-1] > 0:
                y[i] = 0
            elif cur_pct_change[0] < -pct_change_threshold: # and cur_pct_change[-1] < 0:
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
        :param dataToNorm: data to be normalize, size [:, currencies, OHLC]
        :return: normalized data of the same shape as the input
        """

        if self.normalizeTool:
            dataOut = np.zeros(dataToNorm.shape)
            for i in range(dataToNorm.shape[1]):
                    dataOut[:, i, :] = self.normalizeTool[i].fit_transform(dataToNorm[:, i, :])
        else:
            dataOut = dataToNorm

        return dataOut

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
    x = np.array([1, 2, 4, 7, 1])
    gen = FxDataGenerator(x)
    shifted = gen.pct_change_N(x, 2)
    print(shifted)



