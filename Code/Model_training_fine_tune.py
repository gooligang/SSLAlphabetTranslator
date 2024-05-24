import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import time

# Define parameters
IMGSIZE = 224
batch_size = 64
epochs = 10
train_dir = 'TRAINING DIRECTORY'
valid_dir = 'VALIDATION DIRECTORY'
num_classes = 26

# Setting up Tensorboard
NAME = "MobileNet-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

mobile = MobileNet(weights='imagenet')

x = mobile.layers[-5].output
x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
output = Dense(units=26, activation='softmax')(x)

model = Model(inputs = mobile.input, outputs = output)

for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()


train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMGSIZE, IMGSIZE),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMGSIZE, IMGSIZE),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True
)

model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    shuffle = True,
    callbacks = [ModelCheckpoint("CHECKPOINT_MODEL_NAME", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),EarlyStopping(patience=3, monitor='val_loss', mode='min', verbose=1, restore_best_weights=True), tensorboard]
)

test_loss, test_accuracy = model.evaluate(validation_generator)
print("Test accuracy:", test_accuracy)
model.save('MODEL_NAME')
K.clear_session()

