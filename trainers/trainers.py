import os
from torch import Tensor
from typing import Tuple, Union
from data.interfaces import IDataLoader
from .interfaces import ISchedular, ITrainer
from constants import HistoryKeys
from torch.optim import Optimizer
from torch.nn import Module
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel


class BaseTrainer(ITrainer):
    """Builds the basic trainer module

    Args:
        optimizer (Union[Optimizer, ISchedular]): The optimizer or the wrapped
        optimizer that will be used during the training.
        criterion (Module): The loss fucntion that will be used
        during the training process.
        model (Module): The model.
        train_loader (ILoader): The training data loader.
        test_loader (ILoader): The testing data loader.
        epochs (int): The number of epochs.
        log_steps_frequency (int): The number of steps to log the
        results after.
        history (dict): The history of the training if there is
        any. Default {}.
    """
    def __init__(
            self,
            optimizer: Union[Optimizer, ISchedular],
            criterion: Module,
            model: Module,
            train_loader: IDataLoader,
            test_loader: IDataLoader,
            epochs: int,
            log_steps_frequency: int,
            history: dict = {}
            ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.log_steps_frequency = log_steps_frequency
        self.history = history
        if HistoryKeys.test_loss not in self.history:
            self.history[HistoryKeys.test_loss] = list()
        if HistoryKeys.train_loss not in self.history:
            self.history[HistoryKeys.train_loss] = list()
        self.counter = 1

    def backward_pass(self, loss: Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit(self):
        for _ in range(self.epochs):
            self.train()


class BaseDistTrainer(BaseTrainer):
    """Builds the basic distributed data parallel trainer module

    Args:
        optimizer (Union[Optimizer, ISchedular]): The optimizer or the wrapped
        optimizer that will be used during the training.
        criterion (Module): The loss fucntion that will be used
        during the training process.
        model (Module): The model.
        train_loader (ILoader): The training data loader.
        test_loader (ILoader): The testing data loader.
        epochs (int): The number of epochs.
        log_steps_frequency (int): The number of steps to log the
        results after.
        rank (int): The process index.
        world_size (int): The number of nodes/processes.
        dist_address (str): The address of the master node.
        dist_port (int): The port of the master node.
        dist_backend (str): The backend used for DDP.
        history (dict): The history of the training if there is
        any. Default {}.
    """
    def __init__(
            self,
            optimizer: Union[Optimizer, ISchedular],
            criterion: Module,
            model: Module,
            train_loader: IDataLoader,
            test_loader: IDataLoader,
            epochs: int,
            log_steps_frequency: int,
            rank: int,
            world_size: int,
            dist_address: str,
            dist_port: int,
            dist_backend: str,
            history={}
            ) -> None:
        super().__init__(
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            history=history
        )
        self.rank = rank
        self.world_size = world_size
        self.dist_port = dist_port
        self.dist_address = dist_address
        self.dist_backend = dist_backend
        self.init_dist()
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.rank]
        )

    def init_dist(self):
        os.environ['MASTER_ADDR'] = self.dist_address
        os.environ['MASTER_PORT'] = str(self.dist_port)
        init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_address,
            world_size=self.world_size,
            rank=self.rank
        )

    @property
    def is_master(self):
        return self.rank == 0


class CTCTrainer(BaseTrainer):
    def __init__(
            self,
            optimizer: Union[Optimizer, ISchedular],
            criterion: Module,
            model: Module,
            train_loader: IDataLoader,
            test_loader: IDataLoader,
            epochs: int,
            log_steps_frequency: int,
            device: str,
            history: dict = {}
            ) -> None:
        super().__init__(
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            history=history
        )
        self.device = device

    # TODO: add step test decorator
    # TODO: add step log decorator
    def forward_pass(
            self, batch: Tuple[Tensor]) -> Tensor:
        batch = [
            item.to(self.device) for item in batch
            ]
        [speech, speech_mask, text, text_mask] = batch
        preds, lengths = self.model(speech, speech_mask)
        # preds of shape [T, B, C]
        loss = self.criterion(
            preds, text, lengths, text_mask.sum(dim=-1)
            )
        return loss

    # TODO: add train epoch log decorator
    def train(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            loss = self.forward_pass(batch)
            self.backward_pass(loss)
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    # TODO: add epoch test log decorator
    def test(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.test_loader:
            loss = self.forward_pass(batch)
            total_loss += loss
        return total_loss / len(self.test_loader)
