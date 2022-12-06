from abc import ABC, abstractmethod


class IProcessor(ABC):

    @abstractmethod
    def execute():
        pass


class IProcess(ABC):

    @abstractmethod
    def func():
        pass

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
    def tokenize():
        pass
