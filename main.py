import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler

from src.generator import DataGenerator
from src.logger import setup_logger
from src.preprocessing import ProcessData
from src.model_dispatcher import MODEL_DISPATCHER
from src.loss import grad
from src.date_map import get_next_day
from src.trainer import train

LOGGER = setup_logger()  # set up the logger.
scaler = MinMaxScaler()

if __name__ == "__main__":
    LOGGER.info("Loading the training data...")
    train_data = np.load('data/train_data.npy')
    train_data = train_data.item()
    
    final_train_loss_his, final_train_acc_his = [], []
    
    LOGGER.info("Initializing the model")
    LSTM = MODEL_DISPATCHER['LSTM']
    model = LSTM(input_shape=(1,93), output_shape=(1))
    model = model.create_model()
    
    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Number of epochs
    epochs = 2
    
    for key in tqdm(train_data.keys()):
        
        data = train_data[key]
        dataframe = data[0]
        solvency_date = data[1]
        final_timeStep = data[2] * len(dataframe)  # multiply by len to create tensors of equal length for training.
        score = data[3] * len(dataframe)           # multiply by len to create tensors of equal length for training.
        LOGGER.info(f"Total length of training data is {len(dataframe)}")
        
        dataframe = dataframe.drop('Date', axis=1)
        columns = dataframe.columns  # to use in creating float from strings.
        
        # TODO: remove this from the main loop to outside.
        def convert_string_to_float(x):
            for col in columns:
                x[col] = float(str(x[col]).replace(",", "").replace(
                    ' -   ', str(0)).replace("%", ""))
            return x
        
        # apply the convert_string_to_float to get the float from string.
        dataframe = dataframe.apply(convert_string_to_float, axis=1)
        dataframe = dataframe.astype(np.float32)
        normalized_datae = scaler.fit_transform(dataframe.values)
        
        LOGGER.info("Starting training...")
        for epoch in range(epochs):
            # creating data generator.
            generator = DataGenerator(normalized_datae, score, final_timeStep)
            
            # Training
            hist_loss, hist_acc = train(model, epoch, optimizer, grad, generator)
            final_train_loss_his.append(hist_loss)
            final_train_acc_his.append(hist_acc)
            
            