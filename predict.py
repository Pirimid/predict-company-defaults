import os
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

from src.logger import setup_logger
from src.preprocessing import prepare_lstm_data
from src.model_dispatcher import MODEL_DISPATCHER
from src.indicators import enrich_data

LOGGER = setup_logger()  # set up the logger.
scaler = StandardScaler()


# training or testing config
config = {'model_name': 'BiLSTM',
          'data_file': 'train_data',
          'mode': 'train_test',
          'optimizer': tf.keras.optimizers.SGD(learning_rate=0.01),
          'base_name': 'sgd_lr_0.001',
          'time_window': 5,
          'epochs': 1
          }

LOGGER.info("Initializing the model.")
model_init = MODEL_DISPATCHER[config['model_name']]
model = model_init(input_shape=(
    config['time_window'], 326), output_shape=(1))
model = model.create_model()
LOGGER.info("Loading the pretrained model.")
model.load_weights('data/biLstm_5_folds_std_scaler.h5')

LOGGER.info('Loading the data from the database.')
data = np.load(f'data/train_data_v2.npy', allow_pickle=True)
data = data.item()

def make_prediction(user_input):
    """Make prediction for the company name provided.

    Args:
        user_input (str): Name of the company.

    Returns:
        [float]: Predicted score out of 100.
        [float]: Predicted probability of going to default.
    """
    for idx in data.keys():
        company_data = data[idx]
        company = company_data[4]
        company = company.lower()

        if company.find(user_input.lower()) >= 0:
            LOGGER.info(f"Found {user_input} in the database.")
            d = data[idx]
            dataframe = d[0]
            dataframe = dataframe.drop('Date', axis=1)
            # to use in creating float from strings.
            columns = dataframe.columns

            def convert_string_to_float(x):
                for col in columns:
                    x[col] = float(str(x[col]).replace(",", "").replace(
                        ' -   ', str(0)).replace("%", ""))
                return x

            dataframe = dataframe.apply(convert_string_to_float, axis=1)
            dataframe = dataframe.astype(np.float32)
            dataframe = dataframe.drop('index', axis=1)
            dataframe = enrich_data(dataframe)

            if dataframe.shape[0] != 0:
                normalized_data = scaler.fit_transform(dataframe.values)
                lstm_data = prepare_lstm_data(normalized_data, config['time_window'])

                if len(lstm_data.shape) == 3:
                    LOGGER.info("Giving score to the company.")
                    preds = model.predict(lstm_data, batch_size=32)
                    preds = float(preds[-1][0])
                    score = (1- preds) * 100
                    # print("Scrore Predicted: ", round(score, 4), "(Range 0-100)")
                    # print("Probability of going to default: ", round(preds * 100, 4), "%")
                    return round(score, 4), round(preds * 100, 4)

    # if we do not find anything the return None
    return None, None
