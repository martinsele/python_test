from DataReader import DataReader
from FxDataGenerator import FxDataGenerator
from ModelCreator import ModelCreator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import to_categorical
import numpy as np


# pct_change_threshold 0.001 dava pomer zhruba (1:1:2) pro tridy (buy:sell,noop)
# EURCHF se ma minimalni pct_changes (nejcastejsi hodnoty blizke 0.0)
# EURUSD ma celkem dobre pomery pct_change, proto obchodovat zde -> currency_idx = 3
# --> Kontroluje se zmena za periodu (4 hod) ><= 0.001

batch_size = 64
epochs = 20
train_data_ratio = 0.85
time_samples = 24*7
pct_change_threshold = 0.001
currency_idx = 3
rand_seed = 101

model_weights_file = 'cnn_3x3_layers_fake.h5'


def train_model(classes_weights):
    """
    Train model on training data
    :param classes_weight: classes weights to balance classes occurrence
    """
    for e in range(epochs):
        print('Epoch: ', e, '/', epochs)

        # TRAINING
        num_iter = int(data.shape[0] * train_data_ratio / batch_size) + 1
        for i in range(num_iter):
            X, y = generator.generate_rand_train_data(batch_size=batch_size,
                                                      pct_change_threshold=pct_change_threshold,
                                                      currency_idx=currency_idx,
                                                      normalize=True)
            X_transposed = creator.transpose_data(X)
            sample_weights = get_sample_weights(y, classes_weights)
            loss = model.train_on_batch(X_transposed, y, sample_weight=sample_weights)

            # print training error
            print('Iter ', i, '/', num_iter, ' Training loss: ', loss)

        # VALIDATION
        print('Validation:')
        avg_acc = 0
        overall_confmat = np.zeros((3, 3))
        num_iter = int(data.shape[0] * (1 - train_data_ratio) / batch_size) + 1
        for i in range(num_iter):
            X, y = generator.generate_test_data(batch_size=batch_size,
                                                pct_change_threshold=pct_change_threshold,
                                                currency_idx=currency_idx,
                                                normalize=True)
            if len(y) == 0:
                continue
            X_transposed = creator.transpose_data(X)
            y_pred = model.predict_on_batch(X_transposed)
            y_pred_category = np.argmax(y_pred, axis=1)
            y_pred_cat = to_categorical(y_pred_category, num_classes=3)
            # estimate accuracy
            accur, confmat = evaluate_accuracy(y, y_pred_cat)
            overall_confmat += confmat
            avg_acc += accur
            print("Validation", i, "/", num_iter, "  acc:", accur)

        # print validation error
        print("Avg. validation accuracy:", (avg_acc/num_iter))
        print(overall_confmat)

    # save model
    model.save_weights(model_weights_file)


def classify():
    model.load_weights(model_weights_file)

    avg_acc = 0
    overall_confmat = np.zeros((3,3))
    num_iter = int(data.shape[0] * (1 - train_data_ratio) / batch_size) + 1
    for i in range(num_iter):
        X, y = generator.generate_test_data(batch_size=batch_size,
                                            pct_change_threshold=pct_change_threshold,
                                            currency_idx=currency_idx,
                                            normalize=True)
        if len(y) == 0:
            continue

        X_transposed = creator.transpose_data(X)
        y_pred = model.predict_on_batch(X_transposed)
        y_pred_category = np.argmax(y_pred, axis=1)
        y_pred_cat = to_categorical(y_pred_category, num_classes=3)
        # estimate accuracy
        accur, confmat = evaluate_accuracy(y, y_pred_cat)
        overall_confmat += confmat
        avg_acc += accur
        print("Validation", i, "/", num_iter, "  acc:", accur)
        print(confmat)

    # print validation error
    print("Avg. validation accuracy:", (avg_acc / num_iter))
    print(overall_confmat)


def evaluate_accuracy(y_true, y_pred):
    """
    Estimate accuracy of classification
    :param y_true:
    :param y_pred:
    :return: accuracy, confusion matrix
    """
    acc = accuracy_score(y_true, y_pred)
    confmat = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    return acc, confmat


def get_sample_weights(y_true, classes_weights_dict):
    samples = y_true.shape[0]
    samples_w = np.zeros((samples, ))
    for i in range(samples):
        samples_w[i] = classes_weights_dict[np.argmax(y_true[i, :])]
    return samples_w


reader = DataReader('./data')
# data, labels = reader.read_fx_data_as_array()
data, labels = reader.fake_data_as_array()

generator = FxDataGenerator(data,
                            normalizeTool=MinMaxScaler,
                            time_samples=time_samples,
                            trainDataRatio=train_data_ratio,
                            randSeed=rand_seed)


classes_weights = generator.get_classes_bias(pct_change_threshold, currency_idx)

# create model to fit
creator = ModelCreator(data, time_samples, channels_olhc=True)
model = creator.create_cnn_model()
model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',                 # using the Adam optimiser
              metrics=['accuracy'])             # reporting the accuracy

train_model(classes_weights)
# classify()

