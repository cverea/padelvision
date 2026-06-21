"""TrackNet heatmap model (VGG-style encoder/decoder).

Keras is imported inside :func:`build_tracknet` so that importing this module
(and the rest of ``padelvision``) does not require TensorFlow/Keras to be
installed unless ball detection is actually used.
"""
from __future__ import annotations


def build_tracknet(n_classes: int, input_height: int = 360, input_width: int = 640):
    """Build the TrackNet model.

    Input is 3 stacked consecutive RGB frames (9 channels, channels-first);
    output is a per-pixel softmax heatmap with ``n_classes`` bins.
    """
    from keras.layers import (
        Activation,
        BatchNormalization,
        Conv2D,
        Input,
        MaxPooling2D,
        Permute,
        Reshape,
        UpSampling2D,
    )
    from keras.models import Model

    def conv_block(x, filters):
        x = Conv2D(
            filters, (3, 3), kernel_initializer="random_uniform",
            padding="same", data_format="channels_first",
        )(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        return x

    imgs_input = Input(shape=(9, input_height, input_width))

    # Encoder.
    x = conv_block(imgs_input, 64)
    x = conv_block(x, 64)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)

    # Decoder.
    x = UpSampling2D((2, 2), data_format="channels_first")(x)
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    x = UpSampling2D((2, 2), data_format="channels_first")(x)
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = UpSampling2D((2, 2), data_format="channels_first")(x)
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = conv_block(x, n_classes)

    output_shape = Model(imgs_input, x).output_shape
    output_height, output_width = output_shape[2], output_shape[3]

    x = Reshape((-1, output_height * output_width))(x)
    x = Permute((2, 1))(x)
    gaussian_output = Activation("softmax")(x)

    model = Model(imgs_input, gaussian_output)
    model.output_width = output_width
    model.output_height = output_height
    return model
