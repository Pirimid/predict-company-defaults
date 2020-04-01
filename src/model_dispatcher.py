# A simple dispatcher which will dispatch the defined models.

from src.models import LSTMModel
from src.models import DenseModel
from src.models import BiLSTMModel

MODEL_DISPATCHER = {
    'LSTM': LSTMModel,
    'Dense': DenseModel,
    'BiLSTM': BiLSTMModel
}
