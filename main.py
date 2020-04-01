import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from src.generator import DataGenerator
from src.logger import setup_logger
from src.preprocessing import ProcessData, prepare_lstm_data
from src.model_dispatcher import MODEL_DISPATCHER
from src.loss import grad
from src.date_map import get_next_day
from src.trainer import train
from src.plot_training_session import plot_session

LOGGER = setup_logger()  # set up the logger.
scaler = MinMaxScaler()


# training or testing config
config = {'model_name': 'BiLSTM',
          'data_file': 'train_data',
          'mode': 'train_test',
          'optimizer': tf.keras.optimizers.SGD(learning_rate=0.01)}

# main train function.
def trainer(model_name, 
            data_file, 
            optimizer, 
            epochs=2, 
            batch_size=32, 
            test_size=0.2, 
            save_model=True, 
            mode='train_test',
            plot_his=False,
            base_name='baseline'):
    """
     Main train function. 
     
      Arguments:-
      * model_name:- Name of the model from config
      * data_file:- Name of the npy data file to laod
      * optimizer:- Keras optimizer initilized
      * epochs:- Epochs to train the model
      * batch_size:- Batch size to use for training
      * test_size:- test size
      * Save_model:- bool to save the model
      * mode:- Mode to see if it is training or testing model
      * plot_his:- If true then plots the history
      * base_name:- Name to append to the saved model
    """
    LOGGER.info("Loading the training data...")
    train_data = np.load(f'data/{data_file}.npy')
    train_data = train_data.item()
    
    final_train_loss_his, final_train_acc_his = [], []
    
    LOGGER.info("Initializing the model")
    model_init = MODEL_DISPATCHER[model_name]
    model = model_init(input_shape=(5,93), output_shape=(1))
    model = model.create_model()
    
    # compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # keys
    keys = list(train_data.keys())
    # np.random.shuffle(keys) 
    
    test_index = int(len(keys) * test_size)
    train_keys = keys[:len(keys) - test_index]
    test_keys = keys[len(keys) - test_index:]
    
    LOGGER.info(f"Got {len(train_keys)} companies for training...")
    LOGGER.info(f"Got {len(test_keys)} companies for testing...")
    
    if mode == 'train' or mode == 'train_test':
        for key in tqdm(train_keys):
            
            data = train_data[key]
            dataframe = data[0]
            solvency_date = data[1]
            final_timeStep = data[2] * len(dataframe)  # multiply by len to create tensors of equal length for training.
            score = data[3] * len(dataframe)           # multiply by len to create tensors of equal length for training.
            LOGGER.info(f"Total length of training data is {len(dataframe)}")
            LOGGER.info(f"Score : {data[3]}")
            
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
            normalized_data = scaler.fit_transform(dataframe.values)
            lstm_data = prepare_lstm_data(normalized_data)
            
            LOGGER.info("Starting training...")
            model.fit(lstm_data, np.array(score[:-5]), epochs=2, batch_size=32, verbose=2)
        
    if save_model and (mode == 'train_test' or mode == 'train'):
        LOGGER.info("Saving the model now...")
        model.save(f'data/{model_name}_{base_name}.h5')

    if mode == 'test':
        LOGGER.info(f"Loading model {model_name}_{base_name}...")
        model = load_model(f'data/{model_name}_{base_name}.h5')
    
    if mode == 'test' or mode == 'train_test':
        LOGGER.info("Running in testing mode now...")
        true_labels = []
        predictions  = []
        
        for key in tqdm(keys):
            
            data = train_data[test_keys]
            dataframe = data[0]
            solvency_date = data[1]
            final_timeStep = data[2] * len(dataframe)  # multiply by len to create tensors of equal length for training.
            score = data[3] * len(dataframe)           # multiply by len to create tensors of equal length for training.
            LOGGER.info(f"Total length of testing data is {len(dataframe)}")
            LOGGER.info(f"Score : {data[3]}")
            
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
            normalized_data = scaler.fit_transform(dataframe.values)
            
            lstm_data = prepare_lstm_data(normalized_data)

            preds = model.predict(lstm_data, batch_size=32)
            LOGGER.info(f"Accuracy : {accuracy_score(np.array(score[:-5]).reshape(-1,1), preds.round())}")    
            true_labels.extend(np.array(score[:-5]).reshape(-1,1))
            predictions.extend(preds.round())
            
        LOGGER.info(f"Overall accuracy on testing set : {accuracy_score(true_labels, predictions)}")

    if plot_his:
        plot_session(final_train_loss_his, final_train_acc_his)


if __name__ == "__main__":
    trainer(config['model_name'], 
            data_file=config['data_file'],
            optimizer=config['optimizer'], 
            mode=config['mode'])