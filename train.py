import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Grab GPUs that TF found
gpus = tf.config.list_physical_devices('GPU')

if len(gpus) < 1:
    print("WARNING: No GPUs found by Tensorflow")
else:
    print("Succesfully registered GPUs with Tensorflow")

# Download and prep data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
    label_mode='fine'
)
x_train = x_train / 255
x_test = x_test / 255
y_train = tf.keras.utils.to_categorical(
    y_train, num_classes=None, dtype='float32'
)
y_test = tf.keras.utils.to_categorical(
    y_test, num_classes=None, dtype='float32'
)

# Construct dummy model
input_tensor = tf.keras.layers.Input(shape=(32,32,3))
x = MobileNetV2(include_top=False,
                  weights=None,
                  classes=100)(input_tensor)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(100, activation='softmax')(x)
model = tf.keras.Model(inputs=[input_tensor], outputs=[preds])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="categorical_crossentropy",
                           optimizer=optimizer,
                           metrics=['accuracy'])

# Configure data generator and train model
epochs = 10
batch_size = 32

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=5. / 32,
    height_shift_range=5. / 32,
    horizontal_flip=True
)
generator.fit(x_train)
model.fit(generator.flow(x_train, y_train),
                             validation_data=(x_test, y_test),
                             steps_per_epoch=(len(x_train) // batch_size),
                             epochs=epochs, verbose=1,)