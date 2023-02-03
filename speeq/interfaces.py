from abc import ABC, abstractmethod, abstractproperty


class IProcessor(ABC):
    @abstractmethod
    def execute():
        pass


class IProcess(ABC):
    @abstractmethod
    def run():
        pass


class IPadder(ABC):
    @abstractmethod
    def pad():
        pass


class IChecker(ABC):
    @abstractmethod
    def check():
        pass


class IFilter(ABC):
    @abstractmethod
    def filter():
        pass


class ITokenizer(ABC):
    @abstractmethod
    def ids2tokens(self):
        pass

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def set_tokenizer(self):
        pass

    @abstractmethod
    def save_tokenizer(self):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def add_token(self):
        pass

    @abstractmethod
    def preprocess_tokens(self):
        pass

    @abstractmethod
    def batch_tokenizer(self):
        pass

    @abstractproperty
    def vocab_size(self):
        pass

    @abstractmethod
    def get_tokens(self):
        pass


class IDataset(ABC):
    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class IDataLoader(ABC):
    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class ITrainer(ABC):
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def test():
        pass

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def inline_log():
        pass

    @abstractproperty
    def is_master():
        pass


class IScheduler(ABC):
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def step(self):
        pass


class ILogger(ABC):
    @abstractmethod
    def log_step(self):
        pass

    @abstractmethod
    def log(self):
        pass


class ITemplate(ABC):
    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def type(self):
        pass

    @abstractmethod
    def get_dict(self):
        pass
