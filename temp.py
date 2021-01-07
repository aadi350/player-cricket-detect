import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet import preprocess_input
from torchvision_resnet import  decode_segmap
# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ]
)

# Create a model that includes the augmentation stage
input_shape = (224, 224, 3)
classes = 1000
inputs = keras.Input(shape=input_shape)
# # Augment images
# x = data_augmentation(inputs)
# # Rescale image values to [0, 1]
# # Add the rest of the model
x = preprocess_input(inputs)
outputs = keras.applications.ResNet50(weights='imagenet', input_shape=input_shape, classes=classes)(x)
model = keras.Model(inputs, outputs)
model = keras.applications.ResNet50(weights='imagenet', input_shape=input_shape, classes=classes)

img = keras.preprocessing.image.load_img("/media/aadi/Library1/_assets/img/06.jpg", target_size=input_shape)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

print(model.summary())

res = model.predict(img_array)
rgb = decode_segmap(res)
