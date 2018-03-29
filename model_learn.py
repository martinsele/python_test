from DataReader import DataReader
from FxDataGenerator import FxDataGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


# pct_change_threshold 0.001 dava pomer zhruba (1:1:2) pro tridy (buy:sell,noop)
# EURCHF se ma minimalni pct_changes (nejcastejsi hodnoty blizke 0.0)
# EURUSD ma celkem dobre pomery pct_change, proto obchodovat zde -> currency_idx = 3
# --> Kontroluje se zmena za periodu (4 hod) ><= 0.001

batch_size = 64
epochs = 20
train_data_ratio = 0.8
time_samples = 24*14
pct_change_threshold = 0.001
currency_idx = 3
rand_seed = 101


reader = DataReader('./data')
data, labels = reader.read_fx_data_as_array()

generator = FxDataGenerator(data,
                            normalizeTool=MinMaxScaler,
                            time_samples=time_samples,
                            trainDataRatio=train_data_ratio,
                            randSeed=rand_seed)

# create model to fit
# model =
# checkpointer = ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True)
# earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# ---- CHECK THRESHOLD SETTING
classes = np.zeros(3,)
for e in range(epochs):
    print('Epoch: ', e, '/', epochs)

    # TRAINING
    num_iter = int(data.shape[0] * train_data_ratio / batch_size) + 1
    for i in range(num_iter):
        X, y = generator.generate_rand_train_data(batch_size=batch_size,
                                                  pct_change_threshold=pct_change_threshold,
                                                  currency_idx=currency_idx)
        # model.train_on_batch(X, y)

     # print training error

    # VALIDATION
    print('Validation:')
    num_iter = int(data.shape[0] * (1 - train_data_ratio) / batch_size) + 1
    for i in range(num_iter):
        X, y = generator.generate_test_data(batch_size=batch_size,
                                            pct_change_threshold=pct_change_threshold,
                                            currency_idx=currency_idx)
        # y_pred = model.predict_on_batch(X)
        # estimate accuracy

    # print validation error


        # classes[0] += sum(y[:, 0] == 1)
        # classes[1] += sum(y[:, 1] == 1)
        # classes[2] += sum(y[:, 2] == 1)

print(classes)

# ---- CHECK OUTPUT
print(X.shape)
print(y.shape)