import os
import abc

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, RNN, Embedding, Dropout, Flatten, Bidirectional, Conv1D, Add, Multiply, Input
from tensorflow.keras.models import Model
import numpy as np


class BaseModel(abc.ABC):
    """Base class for models."""

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abc.abstractmethod
    def create_model(self):
        pass

    # @abc.abstractmethod
    # def compile(self, **kwargs):
    #     pass


class LSTMModel(BaseModel):
    """
     LSTM model class to create LSTM model.
     Arguments:
      * input_shape: Shape of the input tensor.
      * output_shape: Shape of the output tensor. 
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # model initialization
        self.model = None

    def create_model(self):

        inputs = tf.keras.Input(shape=self.input_shape)

        x = LSTM(64, activation='relu', return_sequences=True)(inputs)
        x = LSTM(64, activation='relu')(x)

        x = Flatten()(x)
        x = Dense(32, activation='elu')(x)
        # x = Dense(32, activation='elu')(x)
        output = Dense(self.output_shape, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=output)
        return self.model


class DenseModel(BaseModel):
    """ 
     A dense model class. 
     Arguments:
      * input_shape: Shape of the input tensor.
      * output_shape: Shape of the output tensor.
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # model initialization
        self.model = None

    def create_model(self):

        inputs = tf.keras.Input(shape=self.input_shape)

        x = Dense(64, activation='elu')(inputs)
        x = Dense(32, activation='elu')(x)
        x = Dense(16, activation='elu')(x)
        output = Dense(self.output_shape, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=output)
        return self.model


class BiLSTMModel(BaseModel):
    """
     Bidirectional LSTM model class to create model.
     
     Arguments:
      * input_shape: Shape of the input tensor.
      * output_shape: Shape of the output tensor. 
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # model initialization
        self.model = None

    def create_model(self):

        inputs = tf.keras.Input(shape=self.input_shape)

        x = Bidirectional(
            LSTM(64, activation='elu', return_sequences=True))(inputs)
        x = Bidirectional(LSTM(64, activation='elu'))(x)

        x = Flatten()(x)
        x = Dense(64, activation='elu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='elu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='elu')(x)
        output = Dense(self.output_shape, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=output)
        return self.model


class waveNet(BaseModel):
    """Wave Net model class. 
    
    Arguments:
      * input_shape: Shape of the input tensor.
      * output_shape: Shape of the output tensor.
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super(waveNet, self).__init__()
        # Config will have all differnet configuration that will be needed in order to create models and train them.
        self.input_shape = input_shape
        self.output_shape = output_shape

    def create_model(self):
        inp = Input(shape=(self.input_shape))
        x = self.wave_block(inp, 16, 3, 8)
        x = self.wave_block(x,  32, 3, 5)
        x = self.wave_block(x,  64, 3, 3)
        x = self.wave_block(x, 128, 3, 1)

        out = Dense(self.output_shape, activation='softmax', name='out')(x)

        model = Model(inputs=inp, outputs=out)

        return model

    def wave_block(self, x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='tanh',
                              dilation_rate=dilation_rate)(x)
            sigm_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='sigmoid',
                              dilation_rate=dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters=filters,
                       kernel_size=1,
                       padding='same')(x)

            res_x = Add()([res_x, x])
            
        return res_x
