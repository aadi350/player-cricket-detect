import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Resizing, RandomCrop
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

# Parameter Inputs
NUM_CLASSES = 2
BATCH_SIZE = 32
image_height = 224
image_width = 224
IMAGE_SIZE = (image_width, image_height)
IMAGE_SIZE_DEPTH = (image_width, image_height, 3)

# data augmentation layers
data_augmentation = tf.keras.Sequential([
	RandomFlip('horizontal'),
	RandomRotation(0.6)
])


def build_model(hparams=None):
	DROPOUT = 0.3
	NUM_UNITS = 32
	if hparams is not None:
		DROPOUT = hparams['HP_DROPOUT']
		NUM_UNITS = hparams['HP_NUM_UNITS']
	# Model configuration
	# feature extraction into

	resize_layer = Resizing(224, 224, name='resize')
	# convert to single 1280 element
	inputs = tf.keras.Input(shape=(None, None, 3))
	x = resize_layer(inputs)
	x = Conv2D(32,3,activation='relu')(x)
	x = GlobalAveragePooling2D()(x)
	x = Flatten()(x)
	x = Dense(32, activation='relu')(x)
	outputs = Dense(NUM_CLASSES, activation='relu')(x)
	model = tf.keras.Model(inputs, outputs)
	print(model.summary())
	return model


def compile_model(model, optimizer=None, loss=None, metrics=None):
	base_learning_rate = 0.0001
	optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(lr=base_learning_rate)
	loss = 'categorical_crossentropy'
	metrics = tfa.metrics.F1Score(
		num_classes=NUM_CLASSES,
		name='f1_score'
	)
	model.compile(optimizer='sgd',
				  loss=loss,
				  metrics=metrics)
	print(model.summary())



def return_model():
	model = build_model()
	compile_model(model)
	return model


if __name__ == '__main__':
	mm = return_model()
