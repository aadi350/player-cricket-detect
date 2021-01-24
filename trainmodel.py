import datetime
import pickle
from models.resnet.model import NUM_CLASSES
import tensorflow as tf
from datetime import datetime
import tensorflow_addons as tfa

# Choosing model
from models.resnet import model

INITIAL_EPOCHS = 250
STEPS_PER_EPOCH = 16
FINE_TUNE_EPOCHS = 10
LOGDIR = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_NAME = model.__name__


def get_callbacks():
	def lr_schedule(epoch):
		"""
      Returns a custom learning rate that decreases as epochs progress.
      """
		learning_rate = 0.2
		if epoch > 10:
			learning_rate = 0.02
		if epoch > 20:
			learning_rate = 0.01
		if epoch > 50:
			learning_rate = 0.005

		tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
		return learning_rate

	checkpoint_filepath = '/tmp/checkpoint'

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)

	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=True,
		monitor=tfa.metrics.F1Score(
			num_classes=NUM_CLASSES,
			name='f1_score'),
		mode='max',
		save_best_only=True)

	return [lr_callback, tensorboard_callback, model_checkpoint_callback]


def train_model(model, train_ds, val_ds):
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		steps_per_epoch=STEPS_PER_EPOCH,
		callbacks=get_callbacks(),
		epochs=INITIAL_EPOCHS
	)

	now = datetime.now().strftime('%d%m%Y_%H%M_')
	model.save('/home/aadi/PycharmProjects/player-cricket-detect/models/saved/' + str(now) + 'mobilenetbatsman.h5')
	return history


def train_model_from_gen(model, train_gen, val_gen):
	history = model.fit(
		train_gen,
		validation_data=val_gen,
		steps_per_epoch=STEPS_PER_EPOCH,
		callbacks=get_callbacks(),
		epochs=INITIAL_EPOCHS,
		validation_steps=8
	)
	pickle.dumps(history)
	now = datetime.now().strftime('%d%m%Y_%H%M_')
	model.save('/home/aadi/PycharmProjects/player-cricket-detect/models/saved/' + str(now) + str(MODEL_NAME) +'.h5')
	return history


if __name__ == '__main__':
	from datagen.dataset_generator import get_datagen

	(train_gen, val_gen), class_indices = get_datagen()
	clf = model.return_model()

	hist = train_model_from_gen(clf, train_gen, val_gen)
