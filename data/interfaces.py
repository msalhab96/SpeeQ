from abc import (
    ABC, abstractmethod, abstractproperty
    )


class IProcessor(ABC):

    @abstractmethod
    def execute():
        pass


class IProcess(ABC):

    @abstractmethod
    def run():
        pass


class IFileLoader(ABC):

    @abstractmethod
    def load():
        pass


class IFormater(ABC):

    @abstractmethod
    def format():
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
