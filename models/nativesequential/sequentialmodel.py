import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.python.keras.callbacks import CSVLogger

from datagen.dataset_generator import get_datagen, get_data
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

IMAGE_SIZE = (224, 224, 3)


def cos_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
    return tf.math.reduce_sum(y_true * y_pred, axis=-1)




sgd_warm_restart = SGD(
    learning_rate=CosineDecayRestarts(0.05, 10)
)


def create_model(optimizer):
    model = Sequential([
        # Conv2D(128, 3, padding='same', input_shape=IMAGE_SIZE, activation='relu'),
        # MaxPooling2D(),
        # Conv2D(64, 3, padding='same', activation='relu'),
        # MaxPooling2D(),
        Conv2D(4, 16, padding='same', activation='relu'),
        Conv2D(4, 8, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=optimizer,
        loss=cos_loss,
        metrics=['accuracy']
    )
    return model


batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learning_rates = [2.5, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
(train_gen, val_gen), train_gen.class_indices = get_datagen()

epoch = 100
batch = 10
lr = 0.01


optimizer = SGD(
    learning_rate=CosineDecayRestarts(lr, 16)
)
model = create_model(optimizer=optimizer)

LOG_PATH = 'savedmodels/cosineloss/b' + str(batch) + 'e' + str(epoch) + 'lr' + str(lr).replace('.','') + 'training.log'
with open(LOG_PATH, 'w', newline='') as csvfile:
    csv_logger = CSVLogger(LOG_PATH)


    model.fit(
        train_gen,
        batch_size=batch_size,
        steps_per_epoch=16,
        epochs=epoch,
        validation_data=val_gen,
        validation_steps=16,
        callbacks=[csv_logger])

    model.save('savedmodels/cosinelossb' + str(batch) + 'e' + str(epoch) + 'lr' + str(lr).replace('.','') + '.h5')

