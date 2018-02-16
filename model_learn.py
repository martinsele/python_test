from DataReader import DataReader
from FxDataGenerator import FxDataGenerator
import numpy as np


reader = DataReader('./data')
data, labels = reader.read_fx_data_as_array()

#
# generator = FxDataGenerator(data, )
#
print(np.array(data).shape)
# print(labels)