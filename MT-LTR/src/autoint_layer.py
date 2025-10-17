#encoding=utf-8
import os
import sys
import argparse
import tensorflow as tf
#import tesorflow_io as tfio
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer, Masking, Lambda, BatchNormalization
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from deep_model.gpu_embedding import GPUEmbedding
# from tensorflow.keras import mixed_precision
# import wandb
# from wandb.keras import WandbMetricsLogger
import json
import bert.tokenization as tokenization
import bert.modeling as modeling
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import tensorflow.compat.v1 as tf_v1


def normalize(inputs, epsilon=1e-8):
    '''
    Applies layer normalization.
    Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    Returns:
        A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** .5)
    outputs = gamma * normalized + beta

    return outputs

class AutoIntLayer(Layer):
    def __init__(self, embedding_size, name='autoint_part'):
        super(AutoIntLayer, self).__init__(name=name)

        self.num_units = embedding_size # d
        activation_function = tf.nn.relu

        # Linear projections
        self.Q = Dense(units=self.num_units, activation=activation_function)
        self.K = Dense(units=self.num_units, activation=activation_function)
        self.V = Dense(units=self.num_units, activation=activation_function)
        self.V_res = Dense(units=self.num_units, activation=activation_function)

        @tf.function
        def call(self, input_emb, training=False, num_head=2, has_residual=True, dropout_keep_prob=0.1):
            # input_emb形状: None * M * d
            # split and concat
            Q_ = tf.concat(tf.split(self.Q(input_emb), num_head, axis=2), axis=0) # None * M * d/num_head
            K_ = tf.concat(tf.split(self.K(input_emb), num_head, axis=2), axis=0) # None * M * d/num_head
            V_ = tf.concat(tf.split(self.V(input_emb), num_head, axis=2), axis=0) # None * M * d/num_head

            # Multiplication
            weights = tf.matmul(Q_, tf.transpose(K_, [0,2,1])) 

            # Scale
            weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

            # Activation
            weights = tf.nn.softmax(weights)

            #Dropout
            if training == False:
                dropout_keep_prob = 0.0
            weights = tf.nn.dropout(weights, dropout_keep_prob)

            # Weighted sum
            outputs = tf.matmul(weights, V_)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_head, axis=0), axis=2)

            # Residual connection
            if has_residual:
                outputs += self.V_res(input_emb)
            outputs = tf.nn.relu(outputs)
            # Normalize
            outputs = normalize(outputs)

            return outputs