# load data using data_loader ~ loader class and directory parametrized
# split data to train/validation/(test?) ~ amount of data in bins parametrized
# normalize data / pct_change
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class FxDataGenerator(object):

    def __init__(self, data, trainDataRatio=0.7, normalizeTool=MinMaxScaler(), randSeed=101):
        '''
        Initialize FX data generator
        :param data: input data of 3 dimensions []
        :param trainDataRatio: split to train and test data - ratio of train/all data
        :param normalizeTool: sklearn's tool for data normalization. By default, use sklearn.preprocessing.MinMaxScaler
        '''
        self.data = data
        self.trainRatio = trainDataRatio

        dataSamples = data.shape[0]
        trainSplit = int(dataSamples*trainDataRatio)


        # self.normalizeTool.fit(data[])
        # self.trainData =








