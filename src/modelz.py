import logging
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from utils import unison_shuffled_copies

from tensorflow import *
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

# tf = tensorflow
tfk = tf.keras
tfkl = tf.keras.layers


def build_BiLSTM_classifier(input_shape, classes=4, seed=420):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=(500, 16), name="Input")

    # Feature extractor
    bilstm = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True))(input_layer)
    bilstm = tfkl.Bidirectional(tfkl.LSTM(64))(bilstm)
    dropout = tfkl.Dropout(0.5, seed=seed)(bilstm)

    # Classifier
    classifier = tfkl.Dense(8, activation="relu")(dropout)
    output_layer = tfkl.Dense(classes, activation="softmax")(classifier)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name="model")

    # Compile the model
    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
        metrics=[F1Score()],
    )

    # Return the model

    return model


def ENGNet2(
    nb_classes,
    Chans=16,
    Samples=500,
    dropoutRate=0.5,
    kernLength=16,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(16, 500, 1))

    # input2 = tfkl.Reshape((16, 500, 1))(input1)
    ##################################################################

    block1 = tfkl.GlobalAveragePooling2D(data_format="channels_first")(input1)
    block2 = tfkl.Reshape((16, 1, 1))(block1)
    block3 = tfkl.Dense(16, activation="sigmoid", name="dense2")(block2)
    concat0 = tfkl.multiply([block3, input1])

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # block2       = SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same') (block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    model = tfk.models.Model(inputs=input1, outputs=softmax)

    # learning_rate = tf.Variable(0.1, trainable=False)
    # tf.keras.backend.set_value(learning_rate, 0.1)

    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=0.0002
        ),  # put this as it should perform better on m1/m2 macs, change to tfk.optimizers.Adam(lr=1) for different architecture
        metrics="accuracy",
    )

    return model


def ENGNet22(
    nb_classes,
    Chans,
    Samples=500,
    dropoutRate=0.5,
    kernLength=16,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, 2500, 1))

    # input2 = tfkl.Reshape((16, 500, 1))(input1)
    ##################################################################

    block1 = tfkl.GlobalAveragePooling2D(data_format="channels_first")(input1)
    block2 = tfkl.Reshape((Chans, 1, 1))(block1)
    block3 = tfkl.Dense(Chans, activation="sigmoid", name="dense2")(block2)
    concat0 = tfkl.multiply([block3, input1])

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
    )(concat0)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # block2       = SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same') (block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    model = tfk.models.Model(inputs=input1, outputs=softmax)

    # learning_rate = tf.Variable(0.1, trainable=False)
    # tf.keras.backend.set_value(learning_rate, 0.1)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=0.0002
        ),  # put this as it should perform better on m1/m2 macs, change to tfk.optimizers.Adam(lr=1) for different architecture
        metrics="accuracy",
    )

    return model


# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name="f1_score", **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name="tp", initializer="zeros")
#         self.false_positives = self.add_weight(name="fp", initializer="zeros")
#         self.false_negatives = self.add_weight(name="fn", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(tf.math.round(y_pred), tf.float32)

#         true_positives = tf.reduce_sum(y_true * y_pred)
#         false_positives = tf.reduce_sum((1 - y_true) * y_pred)
#         false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

#         self.true_positives.assign_add(true_positives)
#         self.false_positives.assign_add(false_positives)
#         self.false_negatives.assign_add(false_negatives)

#     def result(self):
#         precision = self.true_positives / (
#             self.true_positives + self.false_positives + tf.keras.backend.epsilon()
#         )
#         recall = self.true_positives / (
#             self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
#         )
#         f1 = (
#             2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
#         )
#         return f1


def EEGNet10(
    nb_classes,
    Chans,
    Samples=500,
    dropoutRate=0.5,
    kernLength=100,
    F1=16,
    D=2,
    F2=32,
    norm_rate=0.25,
    dropoutType="Dropout",
):

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, 500, 1))
    ##################################################################

    block1 = tfkl.GlobalAveragePooling2D(data_format="channels_first")(input1)
    block2 = tfkl.Reshape((Chans, 1, 1))(block1)
    block3 = tfkl.Dense(Chans, activation="sigmoid", name="dense2")(block2)
    concat0 = tfkl.multiply([block3, input1])

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(concat0)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)

    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, kernLength // 10))(block1)
    block1 = dropoutType(dropoutRate // 2.5)(block1)

    # block2      = SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same') (block1)

    block2 = SeparableConv2D(F2, (1, kernLength // 4), use_bias=False, padding="same")(
        block1
    )

    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, kernLength // 25))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    model = tfk.models.Model(inputs=input1, outputs=softmax)

    # learning_rate = tf.Variable(0.1, trainable=False)
    # tf.keras.backend.set_value(learning_rate, 0.1)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=0.0002
        ),  # put this as it should perform better on m1/m2 macs, change to tfk.optimizers.Adam(lr=1) for different architecture
        metrics="accuracy",
    )

    return model


def EEGNetK50(
    nb_classes,
    Chans,
    Samples=500,
    dropoutRate=0.5,
    kernLength=50,
    F1=16,
    D=2,
    F2=32,
    norm_rate=0.25,
    dropoutType="Dropout",
):

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, Samples, 1))
    ##################################################################

    block1 = tfkl.GlobalAveragePooling2D(data_format="channels_first")(input1)
    block2 = tfkl.Reshape((Chans, 1, 1))(block1)
    block3 = tfkl.Dense(Chans, activation="sigmoid", name="dense2")(block2)
    concat0 = tfkl.multiply([block3, input1])

    ##################################################################

    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(concat0)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)

    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, kernLength // 10))(block1)
    block1 = dropoutType(dropoutRate // 2.5)(block1)

    # block2      = SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same') (block1)

    block2 = SeparableConv2D(F2, (1, kernLength // 4), use_bias=False, padding="same")(
        block1
    )

    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, kernLength // 25))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    ##################################################################

    model = tfk.models.Model(inputs=input1, outputs=softmax)

    # learning_rate = tf.Variable(0.1, trainable=False)
    # tf.keras.backend.set_value(learning_rate, 0.1)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=0.0002
        ),  # put this as it should perform better on m1/m2 macs, change to tfk.optimizers.Adam(lr=1) for different architecture
        metrics="accuracy",
    )

    return model
