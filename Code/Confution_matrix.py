import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.models import load_model


model_name = "C:\\Users\\theoj\\Desktop\\KandidatUpsatsProject\\Exam_project\\Models\\Fine-tuned_23_layers_trainable"
model = tf.keras.models.load_model(model_name)
data_dir = "C:\\Users\\theoj\\Desktop\\PYTHON\\EXAMENS_ARBETE\\examens_arbete\\DATASET_RESHAPED\\TEST\\test_22_04"

image_height = 224
image_width = 224
batch_size = 64

datagen = ImageDataGenerator(rescale=1./255)


data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title = 'Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


y_prob = model.predict(data_generator)
y_pred = np.argmax(y_prob, axis=1)
y_true = data_generator.classes

classes = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

plot_confusion_matrix(y_true, y_pred, classes=classes,title='Confusion matrix, without normalization')
plt.show()

