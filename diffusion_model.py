import tensorflow as tf
from keras.layers import Input, Dense, GlobalMaxPooling2D, UpSampling2D, Concatenate, Conv2D, AveragePooling2D
from tensorflow.keras.models import Model
from network_support import *
from diffusion_tools import *


def create_diffusion_model(image_size,
                           n_channels,
                           n_classes,
                           n_steps,
                           n_embedding,
                           filters,
                           n_conv_per_step=3,
                           conv_activation='elu',
                           kernel_size=3,
                           padding='valid',
                           sdropout=None,
                           batch_normalization=False):
    '''
    Create a generator model
    :param image_size: Size of input image
    :param n_channels: Number of channels in image
    :param n_classes: Number of output classes for image
    :param n_steps: Number of noise steps
    :param n_embedding: Number of embedding dimensions
    :param filters: Filters for each level of down Unet; array
    :param n_conv_per_step: Number of convolutions to perform in each layer of down Unet
    :param conv_activation: Activation for convolution layers
    :param kernel_size: Kernel size for all convolutions
    :param padding: Padding for all convolutions
    :param sdropout: Probability of spatial dropout for all convolutions
    :param batch_normalization: Use batch normalization
    :return: Generator model
    '''
    # Input for labels
    label_input = Input(shape=(image_size[0], image_size[1], n_classes,), name='label_input')

    # Input for noised image
    image_input = Input(shape=(image_size[0], image_size[1], n_channels,), name='image_input')

    # Input for time step
    time_input = Input(shape=(1,), name='time_input', dtype=tf.int32)
    inputs = [label_input, image_input, time_input]

    # Use positional encoding for time step
    time_input = PositionEncoder(max_steps=n_steps, max_dims=n_embedding)(time_input)

    # Broadcast scalar time step to match image size
    time_input = tf.expand_dims(time_input, axis=1)
    time_input = tf.expand_dims(time_input, axis=1)
    time_input = tf.tile(time_input, [1, image_size[0], image_size[1], 1])

    # Concatenate inputs
    time_and_labels = Concatenate(axis=3)([label_input, time_input])
    tensor = Concatenate(axis=3)([time_and_labels, image_input])

    # Scale time and label inputs for skip connections
    time_and_label_skips = []
    for i in range(len(filters)):
        time_and_label_skip = AveragePooling2D(pool_size=(2 ** i, 2 ** i), strides=(2 ** i, 2 ** i))(time_and_labels)
        time_and_label_skips.append(time_and_label_skip)

    # Down convolutions in Unet
    tensor, skips = create_cnn_down_stack(tensor=tensor,
                                          n_conv_per_step=n_conv_per_step,
                                          filters=filters,
                                          activation=conv_activation,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          batch_normalization=batch_normalization,
                                          sdropout=sdropout)

    # Get rid of last skip connection (unneeded)
    skips.pop()

    # Add time and labels in middle of bottom of Unet
    tensor = Concatenate()([tensor, time_and_label_skips.pop()])

    # Up convolutions in Unet
    r_filters = list(reversed(filters))
    for i, f in enumerate(r_filters[:-1]):
        # Finish stack of convolution layers
        tensor = create_conv_stack(tensor=tensor,
                                   n_conv_per_step=n_conv_per_step - 1,
                                   filters=f,
                                   activation=conv_activation,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   batch_normalization=batch_normalization,
                                   sdropout=sdropout)

        # Up sampling
        tensor = UpSampling2D(size=2)(tensor)

        # Concatenate skip connection, time, and labels
        tensor = Concatenate()([tensor, skips.pop(), time_and_label_skips.pop()])

        # First element in next stack of convolution layers, but with more filters
        tensor = create_conv_stack(tensor=tensor,
                                   n_conv_per_step=1,
                                   filters=f,
                                   activation=conv_activation,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   batch_normalization=batch_normalization,
                                   sdropout=sdropout)

    # Finish top layer of Unet
    tensor = create_conv_stack(tensor=tensor,
                               n_conv_per_step=n_conv_per_step - 1,
                               filters=r_filters[-1],
                               activation=conv_activation,
                               kernel_size=kernel_size,
                               padding=padding,
                               batch_normalization=batch_normalization,
                               sdropout=sdropout)

    # Add last convolution to output image
    tensor = Conv2D(filters=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_bias=True,
                    kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='linear')(tensor)
    output = tensor

    # Create model from data flow
    model = Model(inputs=inputs, outputs=output, name='diffusion')

    return model
