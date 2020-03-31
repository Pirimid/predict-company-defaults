# A simple dispatcher which will dispatch the defined models.

from src.models import LSTMModel
from src.models import DenseModel

MODEL_DISPATCHER = {
    'LSTM': LSTMModel,
    'Dense': DenseModel
}
