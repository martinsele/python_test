
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
import numpy as np

class ModelCreator:
    """
    Class for building and handling Keras Model of the data.
    """

    def __init__(self, input_data, time_steps, channels_olhc=False):
        """
        Initialize ModelCreator for particular Keras backend.
        Use different currency data as channels and time_steps x OLHC as width x height of the input
        :param input_data: (all_time_steps, currency, OLHC)
        :param time_steps: number of time_steps in samples
        :param channels_olhc: if true, use OLHC as channels, otherwise use currencies as channels
        """
        input_shape = input_data.shape
        if channels_olhc:
            self.channels = input_shape[2]
            self.width = time_steps
            self.height = input_shape[1]
            if K.image_data_format() == 'channels_first':
                self.input_shape = (self.channels, self.width, self.height)
                # indexes used to transpose batches of shape (batch_size, OLHC, time_steps, currency)
                self.data_transpose_idxs = (0, 3, 1, 2)
            else:
                self.input_shape = (self.width, self.height, self.channels)
                # indexes used to transpose batches of shape (batch_size, time_steps, currency, OLHC)
                self.data_transpose_idxs = (0, 1, 2, 3)

        else:
            self.channels = input_shape[1]
            self.width = time_steps
            self.height = input_shape[2]
            if K.image_data_format() == 'channels_first':
                self.input_shape = (self.channels, self.width, self.height)
                # indexes used to transpose batches of shape (batch_size, currency, time_steps, OLHC)
                self.data_transpose_idxs = (0, 2, 1, 3)
            else:
                self.input_shape = (self.width, self.height, self.channels)
                # indexes used to transpose batches of shape (batch_size, time_steps, OLHC, currency)
                self.data_transpose_idxs = (0, 1, 3, 2)



    def transpose_data(self, data_in):
        """
        Transpose input data to correspond to necessary model structure
        :param data_in: input data
        :return: transposed input data
        """
        data_out = np.transpose(data_in, self.data_transpose_idxs)
        return data_out

    def create_cnn_model(self):
        """
        Create CNN model with convolutional and MaxPooling blocks
        :return: cnn model
        """
        data_input = Input(shape=self.input_shape)      # 336 x 4

        x = Conv2D(32, (1, 5), activation='relu', padding='same')(data_input)
        x = Conv2D(32, (1, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)     # 168 x 4
        # x = Dropout(0.25)(x)

        x = Conv2D(64, (1, 5), activation='relu', padding='same')(x)
        x = Conv2D(64, (1, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D((4, 2), strides=(4, 2), padding='same')(x)     # 42 x 2
        # x = Dropout(0.25)(x)

        x = Conv2D(128, (1, 5), activation='relu', padding='same')(x)
        x = Conv2D(128, (1, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D((4, 2), strides=(4, 2), padding='same')(x)     # 11 x 1
        # x = Dropout(0.25)(x)

        flat = Flatten()(x)
        hidden = Dense(1024, activation='relu', name='dense1')(flat)
        drop_3 = Dropout(0.5)(hidden)
        out = Dense(3, activation='softmax', name='dense2')(drop_3)

        model = Model(inputs=data_input, outputs=out)

        return model





