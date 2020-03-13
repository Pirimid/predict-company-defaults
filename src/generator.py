import tensorflow as tf 
import numpy as np

from src.logger import LOGGER

class DataGenerator:
    """
     A Data generator class that will give the data to feed to network.
     Return:
      {Data}: Feature vector
      {Labels}: Defaulted/Non-defaulted
      {timeStamp}: When did the default occured. If we can predict the default way ahead of time, then model will have less loss. 
    """
    
    def __init__(self, data, labels, timeStamp, **kwargs):
        """
         data: feature vector as list
         labels: Labels as list
         timeStampe: timeStamp as list
        """
        self.data = data
        self.labels = labels
        self.timeStamp = timeStamp
        
        self.features = tf.data.Dataset.from_tensor_slices(self.data)
        self.y = tf.data.Dataset.from_tensor_slices(self.labels)
        self.step = tf.data.Dataset.from_tensor_slices(self.timeStamp)
        
    def get_next_step(self):
        LOGGER.info("Getting next timeStamp data")
        self.data_iterator = iter(self.features)
        self.label_iterator = iter(self.y)
        self.step_iterator = iter(self.step)
        return (next(self.data_iterator).numpy(),  next(self.label_iterator).numpy(), next(self.step_iterator).numpy())