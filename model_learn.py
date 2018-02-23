from DataReader import DataReader
from FxDataGenerator import FxDataGenerator
from sklearn.preprocessing import MinMaxScaler
import numpy as np


batch_size = 10
time_samples = 24*14
pct_change_threshold = 0.1
currency_idx = 0
rand_seed = 101


reader = DataReader('./data')
data, labels = reader.read_fx_data_as_array()


generator = FxDataGenerator(data, normalizeTool=MinMaxScaler, randSeed=rand_seed)

X, y = generator.generate_rand_train_data(time_samples=time_samples,
                                    batch_size=batch_size,
                                    pct_change_threshold=pct_change_threshold,
                                    currency_idx=currency_idx)

print(X.shape)
print(y.shape)
print(y)

# TODO - koukni, jak jsou pct changes rozvrstveny v ramci datasetu pro jednotlive meny