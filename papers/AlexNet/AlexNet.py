from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.Sequential([
    # layer 1
    layers.Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        activation=keras.activations.relu,
        padding='valid',
        input_shape=(227, 227, 3)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    # layer 2
    layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=keras.activations.relu,
        padding='same'
    ),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    # layer 3
    layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=keras.activations.relu,
        padding='same'
    ),
    layers.BatchNormalization(),

    # layer 4
    layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=keras.activations.relu,
        padding='same'
    ),
    layers.BatchNormalization(),

    # layer 5
    layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=keras.activations.relu,
        padding='same'
    ),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    # layer 6
    layers.Flatten(),
    layers.Dense(units=4096, activation=keras.activations.relu),
    layers.Dropout(rate=0.5),

    # layer 7
    layers.Dense(units=4096, activation=keras.activations.relu),
    layers.Dropout(rate=0.5),

    # layer 8
    layers.Dense(units=1000, activation=keras.activations.softmax)
])

model.summary()

# https://www.tensorflow.org/guide/keras/custom_layers_and_models
class AlexNet(keras.Model):
    def __init__(self, num_classes, input_shape=(227, 227, 3)):
        super(AlexNet, self).__init__()
        self.input_layer = layers.Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            activation=keras.activations.relu,
            padding='valid',
            input_shape=input_shape)
        self.middle_layers = [
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation=keras.activations.relu,
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=keras.activations.relu,
                padding='same'
            ),
            layers.BatchNormalization(),

            layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=keras.activations.relu,
                padding='same'
            ),
            layers.BatchNormalization(),

            layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=keras.activations.relu,
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=4096, activation=keras.activations.relu),
            layers.Dropout(rate=0.5),
            layers.Dense(units=4096, activation=keras.activations.relu),
            layers.Dropout(rate=0.5),
        ]
        self.out_layer = layers.Dense(
            units=num_classes, activation=keras.activations.softmax)

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.middle_layers:
            x = layer(x)
        probs = self.out_layer(x)
        return probs

model2 = AlexNet(1000)
model2.build((None, 227, 227, 3))
model2.summary()
