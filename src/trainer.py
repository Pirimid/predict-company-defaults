import os
import time

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pandas as pd

from src.logger import LOGGER

def train(model, epochs, optimizer, grad, train_dataset, curr_timeStep, log_interval):
    """
     A main training function for training the models.
      Arguments:
      * model- Model to train.
      * epochs- number of epochs to train the model
      * optimizer- optimizer to be used.
      * grad- gradient function which computes gradient for the model layers.
      * train_dataset- training dataset on which model is trained.
    """
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 1
        # for i, (x, y, timeStep) in zip(range(0, len(train_dataset[0])), train_dataset):
        x, y, timeStep = train_dataset
        x = x.reshape(1, 1, x.shape[0])
        y = y.reshape(1,1)
        timeStep = timeStep.reshape(1, 1)
        
        # Optimize the model
        loss_value, grads = grad(model, x, y, curr_timeStep, timeStep)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % log_interval == 0:
            LOGGER.info("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
    
    return train_loss_results, train_accuracy_results