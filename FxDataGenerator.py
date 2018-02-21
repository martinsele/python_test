# load data using data_loader ~ loader class and directory parametrized
# split data to train/validation/(test?) ~ amount of data in bins parametrized
# normalize data / pct_change
import numpy as np



class FxDataGenerator(object):

    def __init__(self, data, trainDataRatio=0.8, normalizeTool=None, randSeed=101):
        '''
        Initialize FX data generator
        :param data: input data of 3 dimensions [TimeSteps, currencies, OHLC]
        :param trainDataRatio: split to train and test data - ratio of train/all data
        :param normalizeTool: sklearn's tool for data normalization. E.g., sklearn.preprocessing.MinMaxScaler.
        By default, uses None
        Data are not stationary, thus normalize data locally when producing batches
        '''
        self.data = data
        self.trainRatio = trainDataRatio
        self.randSeed = randSeed

        samples = data.shape[0]
        self.trainSplitIdx = int(samples*trainDataRatio)

        # Prepare normalization of each currency - data not stationary, thus need to normalize only locally
        self.normalizeTool = []
        if normalizeTool:
            for i in range(data.shape[1]):
                self.normalizeTool.append(normalizeTool())
                # self.normalizeTool.append(normalizeTool().fit(data[:self.trainSplitIdx, i, :]))

    def generate_train_data(self, time_samples=24*14, batch_size=10, pct_change_threshold=0.1, currency_idx=0, class_mode='binary', normalize=True):
        '''
        Generate normalized training data samples with labels of X,y
        :param time_samples: number of time steps in X data
        :param batch_size: used batch size
        :param pct_change_threshold: threshold to label dataset as 0 - buy / 1 - sell / 2 - noop
        :param currency_idx: currency in question (buy/sell/nothing)
        :param class_mode: 'binary' for binary labels using one-hot-encode, 'multi' for multiple class labels [0,1,2]
        :param normalize: True if data should be normalized using normalization tool defined in constructor
        :return: X - training data matrix of shape [batch_size, time_samples, currencies, OHCL],
        y - vector or matrix of classes, depending on the parameter 'class_mode'
        '''
        # get random start point within the training data
        rand_start = np.random.randint(0, self.trainSplitIdx-time_samples, (batch_size, ))

        print(rand_start)

        X = np.zeros()
        y = np.zeros()




