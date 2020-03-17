# A simple dispatcher which will dispatch the defined models.

from src.models import LSTMModel

MODEL_DISPATCHER = {
    'LSTM': LSTMModel
}
