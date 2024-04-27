'''
General network building tools.

Andrew H. Fagg

Advanced Machine Learning
'''

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, UpSampling2D, Add
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, \
    Dropout, SpatialDropout2D
from tensorflow.keras import Input, Model
from matplotlib import colors
from tensorflow.keras.utils import plot_model


def create_conv_stack(tensor: tf.Tensor,
                      n_conv_per_step: int,
                      filters: int,
                      kernel_size: int = 3,
                      padding: str = 'same',
                      activation: str = 'elu',
                      **kwargs) -> tf.Tensor:
    '''
    Create a single Conv stack.  Each step has the same number of filters. 
    
    Skip connection from front of the stack to the last Conv layer is optional.
    
    ----> C -> C -> ... -> C ->
       \__________________/
              skip
              
    Each Conv can be preceded with a BN layer, and followed by a spatial dropout layer
    
    Last Conv layer may have a custom activation function
    
    :param tensor: Input tensor (2D data)
    :param step: Index of current stack
    :param n_conv_per_step: Number of convolutional layers in this stack
    :param filters: Number of Conv filters to use
    :param kernel_size: 2D kernel size
    :param padding: Conv padding
    :param activation: Activation function to use
    
    optional kwargs used:
    - conv_skip: the type of skip to use ('C' = concatenation; 'A' = addition)
    - batch_normalization: true -> include batch normalization
    - sdropout': float dropout probability
    - activation_last: activation function used for the last Conv layer
        
    :return: Output tensor 
    '''

    # Remember first tensor
    skip = tensor

    # Loop over each Conv layer in the stack
    for j in range(n_conv_per_step):

        if j == n_conv_per_step - 1:
            # Last Conv layer in stack
            # If specified: include skip connection from before this stack
            if 'conv_skip' in kwargs.keys():
                if kwargs['conv_skip'] == 'C':
                    tensor = Concatenate()([tensor, skip])
                elif kwargs['conv_skip'] == 'A':
                    tensor = Add()([tensor, skip])

        # Optional BN
        if 'batch_normalization' in kwargs.keys() and kwargs['batch_normalization']:
            tensor = BatchNormalization()(tensor)

        # Conv layer
        #  All layers use specified activation function except 
        #  the last may have a custom activation function
        tensor = Convolution2D(filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               use_bias=True,
                               kernel_initializer='random_normal',
                               bias_initializer='zeros',
                               activation=activation if (
                                           j < n_conv_per_step - 1 or not 'activation_last' in kwargs.keys()) else
                               kwargs['activation_last'])(tensor)

        # Optional dropout
        if 'sdropout' in kwargs.keys() and kwargs['sdropout'] is not None:
            tensor = SpatialDropout2D(kwargs['sdropout'])(tensor)

    # Return output
    return tensor


def create_cnn_down_stack(tensor: tf.Tensor,
                          n_conv_per_step: int,
                          filters: [int],
                          activation: str = 'elu',
                          **kwargs) -> (tf.Tensor, [tf.Tensor]):
    '''
    Create a sequence of CNN stacks + down samples.  Specifically, each stack is of
    the form:
        C -> MP -> C -> ... -> C
                \____________/
                    skip
    
    The latter sequence of convultions have an optional skip connection
    
    :param tensor: Input tensor (2D spatial data)
    :param n_conv_per_step: Number of convolutional layers in each step in the stack
    :param filters: Number of filters at each stack level (list of ints)
    :param activation: Default activation function
    
    optional kwargs are used for calls to create_conv_stack()
    
    :return: Output tensor + a list of tensors for skip connections
    '''

    # Track outputs for skip connections
    tensor_list = []

    # First conv stack
    tensor = create_conv_stack(tensor, n_conv_per_step - 1, filters[0], **kwargs)

    print("DOWN STACK: %d" % filters[0])
    for i, f in enumerate(filters[1:]):
        print("DOWN STACK: %d" % f)
        # Last element in the previous conv stack, but we increase the number of filters
        tensor = create_conv_stack(tensor, 1, f,
                                   activation=activation, **kwargs)

        # Add this tensor to the skip connection list
        tensor_list.append(tensor)

        # Max pooling + striding
        tensor = MaxPooling2D(pool_size=2, strides=2)(tensor)

        # Next stack of Conv layers
        tensor = create_conv_stack(tensor, n_conv_per_step - 1, f,
                                   activation=activation, **kwargs)

    # Add last tensor to the list
    tensor_list.append(tensor)

    # Return both the last tensor and the tensors for the skip connections
    return tensor, tensor_list


def create_dense_stack(tensor: tf.Tensor,
                       nhidden: [int],
                       activation='elu',
                       **kwargs) -> tf.Tensor:
    ''' 
    Create a stack of hidden layers
    
    :param tensor: Input tensor
    :param nhidden: List of unit numbers for each layer
    :param activation: activation for each layer
    
    optional kwargs:
    - batch_normalization: True -> use batch normalization
    - dropout: Float dropout probability
    
    :return: Output tensor
    
    '''
    # Loop over layers
    for i, n in enumerate(nhidden):
        # Add BN?
        if 'batch_normalization' in kwargs.keys() and kwargs['batch_normalization']:
            tensor = BatchNormalization()(tensor)

        # Dense layer: assume 'activation' in kwargs
        tensor = Dense(n, activation=activation, use_bias=True,
                       kernel_initializer='random_normal',
                       bias_initializer='zeros', name='D%d' % i)(tensor)

        # Dropout?
        if kwargs['dropout'] is not None:
            tensor = Dropout(kwargs['dropout'])(tensor)

    return tensor
