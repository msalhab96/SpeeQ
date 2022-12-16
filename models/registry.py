from .layers import (
    PackedGRU,
    PackedLSTM,
    PackedRNN
    )


RNN_REGISTRY = {
    'rnn': PackedRNN,
    'lstm': PackedLSTM,
    'gru': PackedGRU
}
