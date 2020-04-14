import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.generator import DataGenerator
from src.logger import setup_logger
from src.preprocessing import ProcessData, prepare_lstm_data
from src.model_dispatcher import MODEL_DISPATCHER
from src.loss import grad
from src.date_map import get_next_day
from src.trainer import train
from src.plot_training_session import plot_session
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

# main train function.


def trainer(model,
            model_name,
            data_file,
            train_data,
            valid_data,
            optimizer,
            epochs=2,
            time_window=5,
            batch_size=32,
            test_size=0.2,
            save_model=False,
            mode='train_test',
            plot_his=False,
            base_name='baseline'):
    """
     Main train function. 

      Arguments:-
      * model_name:- Name of the model from config
      * data_file:- Name of the npy data file to laod
      * train_data:- Training data based on fold
      * valid_data:- Valid data based on fold
      * optimizer:- Keras optimizer initilized
      * epochs:- Epochs to train the model
      * time_window:- Time window for LSTM models
      * batch_size:- Batch size to use for training
      * test_size:- test size
      * Save_model:- bool to save the model
      * mode:- Mode to see if it is training or testing model
      * plot_his:- If true then plots the history
      * base_name:- Name to append to the saved model
    """
    LOGGER.info("Loading the training data...")
    data = np.load(f'data/{data_file}.npy')
    data = data.item()

    LOGGER.info(f"Training with time window of {time_window}...")

    final_train_loss_his, final_train_acc_his = [], []

    # keys
    # keys = list(train_data.keys())
    # np.random.shuffle(keys)

    # test_index = int(len(keys) * test_size)
    # train_keys = keys[:len(keys) - test_index]
    # test_keys = keys[len(keys) - test_index:]

    LOGGER.info(f"Got {len(train_data)} companies for training...")
    LOGGER.info(f"Got {len(valid_data)} companies for testing...")

    if mode == 'train' or mode == 'train_test':
        for key in tqdm(train_data['Keys']):

            d = data[key]
            dataframe = d[0]
            solvency_date = d[1] # multiply by len to create tensors of equal length for training.
            final_timeStep = d[2] * len(dataframe) # multiply by len to create tensors of equal length for training.
            score = d[3] * len(dataframe)
            LOGGER.info(f"Total length of training data is {len(dataframe)}")
            LOGGER.info(f"Score : {d[3]}")

            dataframe = dataframe.drop('Date', axis=1)
            # to use in creating float from strings.
            columns = dataframe.columns

            # TODO: remove this from the main loop to outside.
            def convert_string_to_float(x):
                for col in columns:
                    x[col] = float(str(x[col]).replace(",", "").replace(
                        ' -   ', str(0)).replace("%", ""))
                return x

            # apply the convert_string_to_float to get the float from string.
            dataframe = dataframe.apply(convert_string_to_float, axis=1)
            dataframe = dataframe.astype(np.float32)
            dataframe = dataframe.drop('index', axis=1)
            dataframe = enrich_data(dataframe)
            normalized_data = scaler.fit_transform(dataframe.values)
            lstm_data = prepare_lstm_data(normalized_data, time_window)

            LOGGER.info("Starting training...")
            model.fit(lstm_data, np.array(
                score[:-time_window]), epochs=epochs, batch_size=32, verbose=2)

    if save_model and (mode == 'train_test' or mode == 'train'):
        LOGGER.info("Saving the model now...")
        model.save(f'data/{model_name}_{base_name}.h5')

    if mode == 'test':
        LOGGER.info(f"Loading model {model_name}_{base_name}...")
        model = load_model(f'data/{model_name}_{base_name}.h5')

    if mode == 'test' or mode == 'train_test':
        LOGGER.info("Running in testing mode now...")
        true_labels = []
        predictions = []

        for key in tqdm(valid_data['Keys']):

            d = data[key]
            dataframe = d[0]
            solvency_date = d[1]
            # multiply by len to create tensors of equal length for training.
            final_timeStep = d[2] * len(dataframe)
            # multiply by len to create tensors of equal length for training.
            score = d[3] * len(dataframe)
            LOGGER.info(f"Total length of testing data is {len(dataframe)}")
            LOGGER.info(f"Score : {d[3]}")

            dataframe = dataframe.drop('Date', axis=1)
            # to use in creating float from strings.
            columns = dataframe.columns

            # TODO: remove this from the main loop to outside.
            def convert_string_to_float(x):
                for col in columns:
                    x[col] = float(str(x[col]).replace(",", "").replace(
                        ' -   ', str(0)).replace("%", ""))
                return x

            # apply the convert_string_to_float to get the float from string.
            dataframe = dataframe.apply(convert_string_to_float, axis=1)
            dataframe = dataframe.astype(np.float32)
            dataframe = dataframe.drop('index', axis=1)
            dataframe = enrich_data(dataframe)
            normalized_data = scaler.fit_transform(dataframe.values)

            lstm_data = prepare_lstm_data(normalized_data, time_window)

            preds = model.predict(lstm_data, batch_size=32)
            LOGGER.info(
                f"Accuracy : {accuracy_score(np.array(score[:-time_window]).reshape(-1,1), preds.round())}")
            true_labels.extend(np.array(score[:-time_window]).reshape(-1, 1))
            predictions.extend(preds.round())

        LOGGER.info(
            f"Overall accuracy on testing set : {accuracy_score(true_labels, predictions)}")

    if plot_his:
        plot_session(final_train_loss_his, final_train_acc_his)


if __name__ == "__main__":
    # Read dataframe with keys and score values.
    df = pd.read_csv('data/train_v2.csv')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    LOGGER.info("Initializing the model")
    model_init = MODEL_DISPATCHER[config['model_name']]
    model = model_init(input_shape=(
        config['time_window'], 326), output_shape=(1))
    model = model.create_model()

    # compile the model
    model.compile(optimizer=config['optimizer'],
                  loss='binary_crossentropy', metrics=['accuracy'])

    for train_idx, valid_idx in skf.split(X=df['Keys'], y=df['Scores']):
        train_df = df.loc[train_idx]
        valid_df = df.loc[valid_idx]

        trainer(model,
                config['model_name'],
                data_file=config['data_file'],
                train_data=train_df,
                valid_data=valid_df,
                optimizer=config['optimizer'],
                mode=config['mode'],
                base_name=config['base_name'],
                time_window=config['time_window'],
                epochs=config['epochs']
                )

    # model.save('data/biLstm_5_folds_std_scaler.h5')
