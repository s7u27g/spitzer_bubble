import pathlib
import numpy as np
import pandas as pd
import pickle
from tensorflow import  keras

from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from ..utils import file_utils


def get_cnn():
    num_classes = 2
    img_rows = 50
    img_cols = 50
    input_shape = (img_rows, img_cols, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax' , kernel_regularizer=keras.regularizers.l2(0.1)))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['acc'],
    )

    return model

def initialize_learning(ds_path):
    return Learning(ds_path)


class Learning(object):

    def __init__(self, ds_path):
        self.batch_size = 2048
        self.epochs = 500
        self.model = get_cnn()
        self.open_dataset(ds_path)
        pass

    def open_dataset(self, ds_path):
        self.x_train = np.load(ds_path/'x_train.npy')[:,:,:,0:2]
        self.x_test = np.load(ds_path/'x_test.npy')[:,:,:,0:2]
        self.y_train = np.load(ds_path/'y_train.npy')
        self.y_test = np.load(ds_path/'y_test.npy')
        pass

    def fit_model(self):
        es_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=70,
            verbose=1,
            mode='auto'
        )

        rd_cb = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            verbose=1,
            min_lr=0.001,
            mode='auto'
        )

        cbks = [es_cb, rd_cb]
        self.history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=cbks,
            validation_data=(self.x_test, self.y_test)
        )
        pass

    def save_model(self, save_path):
        self.model.save(save_path)
        pass

    def save_history(self, save_path):
#         learn_history = {
#             'acc': np.array(self.history.history['acc']),
#             'val_acc': np.array(self.history.history['val_acc']),
#             'loss': np.array(self.history.history['loss']),
#             'val_loss': np.array(self.history.history['val_loss']),
#         }
        learn_history = pd.DataFrame(self.history.history).to_dict('records')
        file_utils.save_json(save_path, learn_history)
        pass
