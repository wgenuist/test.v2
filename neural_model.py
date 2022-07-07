import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.models import Model


def network_seq2point(seqlen):
    input_layer = Input(shape=seqlen)
    reshape_layer = Reshape((1, seqlen, 1))(input_layer)
    conv_layer_1 = Conv1D(filters=30, kernel_size=10, strides=1, padding='same',
                    activation='relu')(reshape_layer)
    conv_layer_2 = Conv1D(filters=30, kernel_size=8, strides=1, padding='same',
                    activation='relu')(conv_layer_1)
    conv_layer_3 = Conv1D(filters=40, kernel_size=6, strides=1, padding='same',
                    activation='relu')(conv_layer_2)
    conv_layer_4 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same',
                    activation='relu')(conv_layer_3)
    conv_layer_5 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same',
                    activation='relu')(conv_layer_4)
    flatten_layer = Flatten()(conv_layer_5)
    label_layer = Dense(1024, activation='relu')(flatten_layer)
    output_layer = Dense(1, activation='linear')(label_layer)
    model = Model(inputs=input_layer, outputs=output_layer, name='Seq2Point')
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def seq2point_reduced(seqlen):
    """ Adapted from : https://github.com/JackBarber98/pruned-nilm/blob/master/model_structure.py#L56
    """
    input_layer = Input(shape=seqlen)
    reshape_layer = Reshape((1, seqlen, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=20, kernel_size=(8, 1), strides=(1, 1), padding="same",
                                                 activation="relu")(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=20, kernel_size=(6, 1), strides=(1, 1), padding="same",
                                                 activation="relu")(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                                 activation="relu")(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(4, 1), strides=(1, 1), padding="same",
                                                 activation="relu")(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(4, 1), strides=(1, 1), padding="same",
                                                 activation="relu")(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(512, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)
    model = Model(inputs=input_layer, outputs=output_layer, name='Seq2Point')
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
