from DataReader import DataReader
from FxDataGenerator import FxDataGenerator
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# pct_change_threshold 0.001 dava pomer zhruba (1:1:2) pro tridy (buy:sell,noop)
# EURCHF se ma minimalni pct_changes (nejcastejsi hodnoty blizke 0.0)
# EURUSD ma celkem dobre pomery pct_change, proto obchodovat zde -> currency_idx = 3
# --> Kontroluje se zmena za periodu (4 hod) ><= 0.001

batch_size = 10
time_samples = 24*14
pct_change_threshold = 0.001
currency_idx = 3
rand_seed = 101


reader = DataReader('./data')
data, labels = reader.read_fx_data_as_array()

generator = FxDataGenerator(data, normalizeTool=MinMaxScaler, randSeed=rand_seed)

# ---- CHECK THRESHOLD SETTING
classes = np.zeros(3,)
for i in range(400):
    X, y = generator.generate_rand_train_data(time_samples=time_samples,
                                    batch_size=batch_size,
                                    pct_change_threshold=pct_change_threshold,
                                    currency_idx=currency_idx)
    classes[0] += sum(y[:, 0] == 1)
    classes[1] += sum(y[:, 1] == 1)
    classes[2] += sum(y[:, 2] == 1)

print(classes)

# ---- CHECK OUTPUT
print(X.shape)
print(y.shape)

# TODO - koukni, jak jsou pct changes rozvrstveny v ramci datasetu pro jednotlive meny