import numpy as np
import pickle
import cv2, os
import tensorflow as tf
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import DepthwiseConv2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization,ReLU
from keras.callbacks import ModelCheckpoint, Callback,EarlyStopping
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import backend as K
import keras.preprocessing.image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tensorboard = TensorBoard(log_dir= f'logs/model10_grayscale')
IMGSIZE = 128
batch_size = 64
epochs = 30

# lr_decays: 5e-4 2.5e-4 1.25e-4 6.25e-5
class LRAdjustOnPlateau(Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=3, min_lr=5e-5):
        super(LRAdjustOnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.num_increases = 0
        self.best_loss = None
        self.current_lr = None

    def on_train_begin(self, logs=None):
        self.current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss:
            self.num_increases += 1
        else:
            self.num_increases = 0
            self.best_loss = current_loss

        if self.num_increases >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f'\nLearning rate reduced to {new_lr} after {self.patience} consecutive increases in {self.monitor}.')
            self.num_increases = 0
            self.current_lr = new_lr


def get_num_of_classes():
    return 26  # len(glob('gestures/*'))



train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\theoj\\Desktop\\PYTHON\EXAMENS_ARBETE\\examens_arbete\\Dataset_agumentations\\Training\\Training_24',  # Path to the training directory
    target_size=(IMGSIZE, IMGSIZE),  # Resize images to 224x224
    batch_size=batch_size,  # Batch size
    class_mode='categorical',  # Since you have categorical labels
    shuffle =True
)

valid_generator = valid_datagen.flow_from_directory(
    'C:\\Users\\theoj\Desktop\\PYTHON\\EXAMENS_ARBETE\\examens_arbete\\Dataset_agumentations\\Validation\\Validation_24',  # Path to the validation directory
    target_size=(IMGSIZE, IMGSIZE),  # Resize images to 224x224
    batch_size=batch_size,  # Batch size
    class_mode='categorical',  # Since you have categorical labels
    shuffle = True
)


def cnn_model_grayscale():
    num_of_classes = get_num_of_classes()
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), input_shape=(IMGSIZE, IMGSIZE, 1)))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (1, 1), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (1, 1), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1)))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1)))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())


    model.add(DepthwiseConv2D(3, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=512, kernel_size=3, strides=(1, 1)))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "NAME_TO_SAVE_BEST_VERSION"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [checkpoint1,early_stopping_monitor,lr_adjust_callback]
    return model, callbacks_list



def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), input_shape=(IMGSIZE, IMGSIZE, 3)))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (1, 1), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (1, 1), padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same'))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1)))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(DepthwiseConv2D(3, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1)))
    model.add(Dropout(0.10))
    model.add(BatchNormalization())  # ---
    model.add(ReLU())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "NAME_TO_SAVE_BEST_VERSION"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [checkpoint1,early_stopping_monitor,lr_adjust_callback]
    return model, callbacks_list


def train():
    model, callbacks_list = cnn_model_grayscale()
    model.summary()
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,  # Number of steps per epoch
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // batch_size,  # Number of validation steps
        epochs=epochs
        , shuffle=True,
        callbacks=callbacks_list
    )
    model.save('NAME_TO_SAVE_MODEL_AS')


lr_adjust_callback = LRAdjustOnPlateau(patience=2)
early_stopping_monitor = EarlyStopping(patience=4, monitor='val_loss', mode='min', verbose=1)
train()
K.clear_session()






