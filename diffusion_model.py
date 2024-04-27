import tensorflow as tf
from keras.layers import Input, Dense, GlobalMaxPooling2D, UpSampling2D, Concatenate, Conv2D
from tensorflow.keras.models import Model
from network_support import *


def create_discriminator(image_size,
                         n_channels,
                         n_classes,
                         filters,
                         hidden,
                         n_conv_per_step=3,
                         conv_activation='elu',
                         kernel_size=3,
                         padding='valid',
                         sdropout=None,
                         dense_activation='elu',
                         dropout=None,
                         batch_normalization=False):
    '''
    Creates a discriminator model
    :param image_size: Size of input image
    :param n_channels: Number of channels in image
    :param n_classes: Number of output classes for image
    :param filters: Filters for each level of down Unet; array
    :param hidden: Number of neurons in each hidden layer after convolutions; array
    :param n_conv_per_step: Number of convolutions to perform in each layer of down Unet
    :param conv_activation: Activation for convolution layers
    :param kernel_size: Kernel size for all convolutions
    :param padding: Padding for all convolutions
    :param sdropout: Probability of spatial dropout for all convolutions
    :param dense_activation: Activation function for hidden layers
    :param dropout: Probability of dropout for hidden layers
    :param batch_normalization: Use batch normalization
    :return: Discriminator model
    '''
    # Input for RGB image
    input1 = Input(shape=(image_size[0], image_size[1], n_channels,), name='RGB_Image')

    # Input for semantic labels of pixels
    input2 = Input(shape=(image_size[0], image_size[1], n_classes,), name='Pixel_Labels')

    # Concatenate inputs
    tensor = Concatenate()([input1, input2])

    # Create down side of Unet
    tensor, _ = create_cnn_down_stack(tensor=tensor,
                                      n_conv_per_step=n_conv_per_step,
                                      filters=filters,
                                      activation=conv_activation,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      batch_normalization=batch_normalization,
                                      sdropout=sdropout)

    # Global max pooling for each filter
    tensor = GlobalMaxPooling2D()(tensor)

    # Create dense layers
    tensor = create_dense_stack(tensor=tensor,
                                nhidden=hidden,
                                activation=dense_activation,
                                batch_normalization=batch_normalization,
                                dropout=dropout)

    # Single output for if the image is real or fake
    tensor = Dense(1, activation='sigmoid')(tensor)
    output = tensor

    # Create model from data flow
    model = Model(inputs=[input1, input2], outputs=output, name='Discriminator')

    return model


def create_generator(image_size,
                     n_channels,
                     n_classes,
                     n_noise_steps,
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
    :param n_noise_steps: Number of noise vectors to take as input
    :param filters: Filters for each level of down Unet; array
    :param n_conv_per_step: Number of convolutions to perform in each layer of down Unet
    :param conv_activation: Activation for convolution layers
    :param kernel_size: Kernel size for all convolutions
    :param padding: Padding for all convolutions
    :param sdropout: Probability of spatial dropout for all convolutions
    :param batch_normalization: Use batch normalization
    :return: Generator model
    '''
    # Input image with semantic pixel labels
    tensor = Input(shape=(image_size[0], image_size[1], n_classes,), name='Pixel_Labels')
    inputs = [tensor]

    # Input noises at each level in Unet
    noises = []
    for i in range(n_noise_steps - 1, -1, -1):
        noise = Input(shape=(image_size[0] // (2 ** i), image_size[1] // (2 ** i), 1,), name=f'Noise_{i}')
        inputs.append(noise)
        noises.append(noise)

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

    # Add noise in middle of bottom of Unet
    tensor = Concatenate()([tensor, noises.pop(0)])

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

        # Concatenate skip connection and noise
        tensor = Concatenate()([tensor, skips.pop(), noises.pop(0)])

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
                    activation='sigmoid')(tensor)
    output = tensor

    # Create model from data flow
    model = Model(inputs=inputs, outputs=output, name='Generator')

    return model


def create_gan(image_size,
               n_channels,
               n_classes,
               d_filters,
               d_hidden,
               g_n_noise_steps,
               g_filters,
               d_n_conv_per_step=3,
               d_conv_activation='elu',
               d_kernel_size=3,
               d_padding='valid',
               d_sdropout=None,
               d_dense_activation='elu',
               d_dropout=None,
               d_batch_normalization=False,
               d_lrate=0.0001,
               d_loss=None,
               d_metrics=None,
               g_n_conv_per_step=3,
               g_conv_activation='elu',
               g_kernel_size=3,
               g_padding='valid',
               g_sdropout=None,
               g_batch_normalization=False,
               m_lrate=0.0001,
               m_loss=None,
               m_metrics=None):
    '''
    Creates a meta model to train the generator model
    :param image_size: Size of input image
    :param n_channels: Number of channels in image
    :param n_classes: Number of output classes for image
    :param d_filters: Filters for each level of down Unet in discriminator; array
    :param g_n_noise_steps: Number of noise vectors to take as input in generator
    :param g_filters: Filters for each level of down Unet in generator; array
    :param d_hidden: Number of neurons in each hidden layer after convolutions in discriminator; array
    :param d_n_conv_per_step: Number of convolutions to perform in each layer of down Unet in discriminator
    :param d_conv_activation: Activation for convolution layers in discriminator
    :param d_kernel_size: Kernel size for all convolutions in discriminator
    :param d_padding: Padding for all convolutions in discriminator
    :param d_sdropout: Probability of spatial dropout for all convolutions in discriminator
    :param d_dense_activation: Activation function for hidden layers in discriminator
    :param d_dropout: Probability of dropout for hidden layers in discriminator
    :param d_batch_normalization: Use batch normalization in discriminator
    :param d_lrate: Learning rate for discriminator
    :param d_loss: Loss for discriminator
    :param d_metrics: Metrics for discriminator
    :param g_n_conv_per_step: Number of convolutions to perform in each layer of down Unet in generator
    :param g_conv_activation: Activation for convolution layers in generator
    :param g_kernel_size: Kernel size for all convolutions in generator
    :param g_padding: Padding for all convolutions in generator
    :param g_sdropout: Probability of spatial dropout for all convolutions in generator
    :param g_batch_normalization: Use batch normalization in generator
    :param m_lrate: Learning rate for generator
    :param m_loss: Loss for generator
    :param m_metrics: Metrics for generator
    :return:
    '''
    # Create discriminator based on inputs
    d = create_discriminator(image_size=image_size,
                             n_channels=n_channels,
                             n_classes=n_classes,
                             filters=d_filters,
                             hidden=d_hidden,
                             n_conv_per_step=d_n_conv_per_step,
                             conv_activation=d_conv_activation,
                             kernel_size=d_kernel_size,
                             padding=d_padding,
                             sdropout=d_sdropout,
                             dense_activation=d_dense_activation,
                             dropout=d_dropout,
                             batch_normalization=d_batch_normalization)

    # Compile discriminator
    d_opt = tf.keras.optimizers.Adam(learning_rate=d_lrate, amsgrad=False)
    d.compile(loss=d_loss, optimizer=d_opt, metrics=d_metrics)

    # Create generator based on inputs
    g = create_generator(image_size=image_size,
                         n_channels=n_channels,
                         n_classes=n_classes,
                         n_noise_steps=g_n_noise_steps,
                         filters=g_filters,
                         n_conv_per_step=g_n_conv_per_step,
                         conv_activation=g_conv_activation,
                         kernel_size=g_kernel_size,
                         padding=g_padding,
                         sdropout=g_sdropout,
                         batch_normalization=g_batch_normalization)

    # Make discriminator untrainable in meta model
    d.trainable = False

    # Create inputs for the meta model
    labels = Input(shape=(image_size[0], image_size[1], n_classes,), name='Pixel_Labels')
    inputs = [labels]

    # Input noises at each level in Unet
    for i in range(g_n_noise_steps - 1, -1, -1):
        noise = Input(shape=(image_size[0] // (2 ** i), image_size[1] // (2 ** i), 1,), name=f'Noise_{i}')
        inputs.append(noise)

    # Create fake image using generator with labels and noise
    fake_image = g(inputs)

    # Pass fake image and labels to discriminator to see if it is fooled
    p_fake = d([fake_image, labels])

    # Create the meta model
    model = Model(inputs=inputs, outputs=p_fake, name='Meta_Model')

    # Compile meta model
    m_opt = tf.keras.optimizers.Adam(learning_rate=m_lrate, amsgrad=False)
    model.compile(loss=m_loss, optimizer=m_opt, metrics=m_metrics)

    return d, g, model
