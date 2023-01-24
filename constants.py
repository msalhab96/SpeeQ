from enum import Enum

OPTIMIZER_STATE_KEY = 'optimizer'
SCHEDULER_TYPE_KEY = 'scheduler'
CTC_TYPE = 'ctc'
SEQ2SEQ_TYPE = 'seq2seq'
TRANSDUCER_TYPE = 'transducer'


class FileKeys(Enum):
    text_key = 'text'
    speech_key = 'file_path'
    duration_key = 'duration'


class StateKeys(Enum):
    history = 'history'
    epoch = 'epoch'
    model = 'model'
    step = 'step'
    optimizer = 'optimizer'


class HistoryKeys(Enum):
    train_loss = 'train_loss'
    test_loss = 'test_loss'


class LogCategories(Enum):
    batches = 'batches'
    steps = 'steps'
    epochs = 'epochs'
