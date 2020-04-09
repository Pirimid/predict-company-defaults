# A simple dispatcher which will dispatch the defined models.

from src.models import LSTMModel, DenseModel, BiLSTMModel, waveNet

MODEL_DISPATCHER = {
    'LSTM': LSTMModel,
    'Dense': DenseModel,
    'BiLSTM': BiLSTMModel,
    'waveNet': waveNet
}
