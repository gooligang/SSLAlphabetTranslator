from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
import time

# Define parameters
IMGSIZE = 224
batch_size = 64
epochs = 5
num_classes = 26
train_data_dir = 'C:\\Users\\theoj\\Desktop\\PYTHON\\EXAMENS_ARBETE\\examens_arbete\\DATASET_RESHAPED\\TRAIN\\train_22_04'
validation_data_dir = 'C:\\Users\\theoj\\Desktop\\PYTHON\\EXAMENS_ARBETE\\examens_arbete\\DATASET_RESHAPED\\VALIDATION\\VAL_22_04'

NAME = "MobileNetDropout-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


# Load MobileNet model
mobile = MobileNet(weights='imagenet')

# Freeze layers except the last 23 layers
for layer in mobile.layers[:-23]:
    layer.trainable = False

# Add new classification layers after GlobalAveragePooling2D
x = mobile.layers[-6].output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(units=num_classes, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMGSIZE, IMGSIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMGSIZE, IMGSIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    shuffle=True,
    callbacks=[
        ModelCheckpoint("CHECKPOINT_MODEL_NAME", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(patience=3, monitor='val_loss', mode='min', verbose=1, restore_best_weights=True),
        tensorboard
    ]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(validation_generator)
print("Test accuracy:", test_accuracy)
model.save('MODEL_NAME')
K.clear_session()