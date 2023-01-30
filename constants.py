from enum import Enum

# %% Tempaltes types
CTC_TYPE = 'ctc'
SEQ2SEQ_TYPE = 'seq2seq'
TRANSDUCER_TYPE = 'transducer'
MODEL_BUILDER_TYPE = 'model_builder'
# %%

# %% constant keys
OPTIMIZER_STATE_KEY = 'optimizer'
SCHEDULER_TYPE_KEY = 'scheduler'
HIDDEN_STATE_KEY = 'h'
ENC_OUT_KEY = 'enc_out'
PREDS_KEY = 'preds'
PROBABILITIES_KEY = 'props'
TERMINATION_STATE_KEY = 'is_term'
SPEECH_IDX_KEY = 'speech_idx'
DECODER_OUT_KEY = 'decoder_out'
PREV_HIDDEN_STATE_KEY = 'prev_h'
# %%


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
