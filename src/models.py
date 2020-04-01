import os
import abc 

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, RNN, Embedding, Dropout, Flatten, Bidirectional
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
        
        self.model = Model(inputs= inputs, outputs= output)
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
        
        self.model = Model(inputs= inputs, outputs= output)
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
        
        x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(inputs)
        x = Bidirectional(LSTM(64, activation='relu'))(x)
        
        x = Flatten()(x)
        x = Dense(32, activation='elu')(x)
        x = Dense(32, activation='elu')(x)
        output = Dense(self.output_shape, activation='sigmoid')(x)
        
        self.model = Model(inputs= inputs, outputs= output)
        return self.model