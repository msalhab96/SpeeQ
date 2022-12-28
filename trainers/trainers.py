import os
from data.factories import get_asr_loaders, get_tokenizer
from models.registry import get_model
import torch
from torch import Tensor
from typing import Tuple, Union
from trainers.decorators import step_log
from trainers.registry import get_optimizer
from trainers.registry import get_criterion
from utils.utils import set_state_dict
from utils.loggers import ILogger, get_logger
from interfaces import (
    ISchedular, ITrainer, IDataLoader
    )
from constants import HistoryKeys, LogCategories
from torch.optim import Optimizer
from torch.nn import Module
from torch.distributed import (
    init_process_group, barrier, ReduceOp, all_reduce
    )
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn
from tqdm import tqdm


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
        logger (ILogger): The logger to be used.
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
            logger: ILogger,
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
        self.logger = logger
        self.history = history
        self.counter = 1

    def backward_pass(self, loss: Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit(self):
        for _ in range(self.epochs):
            self.train()
            self.logger.log(self.history)

    def inline_log(self, key: str, category: str, value: int):
        tag = f'{key}_{category}'
        if tag in self.history:
            self.history[tag].append(value)
        else:
            self.history[tag] = [value]
        self.logger.log_step(key, category, value)


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
        logger (ILogger): The logger to be used.
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
            logger: ILogger,
            rank: int,
            world_size: int,
            dist_address: str,
            dist_port: int,
            dist_backend: str,
            history={}
            ) -> None:
        BaseTrainer.__init__(
            self,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            logger=logger,
            history=history
        )
        self.rank = rank
        self.world_size = world_size
        self.dist_port = dist_port
        self.dist_address = dist_address
        self.dist_backend = dist_backend
        self.init_dist()
        self.model.to(f'cuda:{rank}')
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

    def _all_reduce_loss(
            self, total_loss: float, counter: int
            ) -> Tensor:
        total = torch.tensor([total_loss/counter]).cuda(self.rank)
        all_reduce(total, op=ReduceOp.SUM)
        return total / self.world_size


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
            logger: ILogger,
            history: dict = {}
            ) -> None:
        BaseTrainer.__init__(
            self,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            logger=logger,
            history=history
        )
        self.device = device

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

    @property
    def is_master(self):
        return True

    @step_log(
        key=HistoryKeys.train_loss.value,
        category=LogCategories.batches.value
        )
    def train_step(self, batch: Tuple[Tensor]) -> float:
        loss = self.forward_pass(batch)
        self.backward_pass(loss)
        return loss.item()

    @step_log(
        key=HistoryKeys.train_loss.value,
        category=LogCategories.epochs.value
        )
    def train(self) -> float:
        self.model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(self.train_loader)):
            loss = self.train_step(batch)
            total_loss += loss
            if (i + 1) % self.log_steps_frequency == 0:
                self.test()
                self.model.train()
                self.inline_log(
                    key=HistoryKeys.train_loss.value,
                    category=LogCategories.steps.value,
                    value=total_loss / (i + 1)
                    )
        return total_loss / len(self.train_loader)

    @step_log(
        key=HistoryKeys.test_loss.value,
        category=LogCategories.steps.value
        )
    @torch.no_grad()
    def test(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.test_loader:
            loss = self.forward_pass(batch)
            total_loss += loss.item()
        return total_loss / len(self.test_loader)


class DistCTCTrainer(BaseDistTrainer, CTCTrainer):
    def __init__(
            self,
            optimizer: Union[Optimizer, ISchedular],
            criterion: Module,
            model: Module,
            train_loader: IDataLoader,
            test_loader: IDataLoader,
            epochs: int,
            logger: ILogger,
            log_steps_frequency: int,
            rank: int,
            world_size: int,
            dist_address: int,
            dist_port: int,
            dist_backend: str,
            history: dict = {}
            ) -> None:
        CTCTrainer.__init__(
            self,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            device=f'cuda:{rank}',
            logger=logger,
            history=history
        )
        BaseDistTrainer.__init__(
            self,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            logger=logger,
            rank=rank,
            world_size=world_size,
            dist_address=dist_address,
            dist_port=dist_port,
            dist_backend=dist_backend,
            history=history
        )

    @step_log(
        key=HistoryKeys.train_loss.value,
        category=LogCategories.epochs.value
        )
    def train(self) -> float:
        self.model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(self.train_loader)):
            loss = self.train_step(batch)
            total_loss += loss
            if (i + 1) % self.log_steps_frequency == 0:
                total = self._all_reduce_loss(total_loss, i + 1)
                if self.is_master():
                    self.inline_log(
                        key=HistoryKeys.train_loss.value,
                        category=LogCategories.steps.value,
                        value=total.item()
                        )
                    self.testdist_config()
                barrier()
        return self._all_reduce_loss(total_loss, len(self.train_loader)).item()

    def fit(self):
        for _ in range(self.epochs):
            self.train()
            if self.is_master:
                self.logger.log(self.history)
            barrier()


def run_asr_trainer(
        rank: int,
        world_size: int,
        trainer_config,
        data_config,
        model_config
        ):
    # TODO: Refactor this code
    if rank > 0:
        import time
        time.sleep(2)
    logger = get_logger(
        name=trainer_config.logger,
        log_dir=trainer_config.logdir,
        n_logs=trainer_config.n_logs,
        clear_screen=trainer_config.clear_screen
        )
    tokenizer = get_tokenizer(
        data_config=data_config
        )
    is_ctc = model_config.template._type == 'ctc'
    model = get_model(
        model_config,
        n_classes=tokenizer.vocab_size
        )
    model = model.to(trainer_config.device)
    optimizer = get_optimizer(model, trainer_config)
    criterion = get_criterion(
        name=trainer_config.criterion,
        blank_id=tokenizer.special_tokens.blank_id,
        pad_id=tokenizer.special_tokens.pad_id,
        **trainer_config.criterion_args
    )
    if os.path.exists(model_config.model_path):
        epoch, steps, history = set_state_dict(
            model=model,
            optimizer=optimizer,
            state_path=model_config.model_path
            )
    else:
        history = {}
    train_loader, test_loader = get_asr_loaders(
        data_config=data_config,
        is_ctc=is_ctc,
        tokenizer=tokenizer,
        batch_size=trainer_config.batch_size,
        world_size=world_size,
        rank=rank
    )
    if world_size > 1:
        if is_ctc:
            trainer = DistCTCTrainer(
                optimizer=optimizer,
                criterion=criterion,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=trainer_config.epochs,
                log_steps_frequency=trainer_config.log_steps_frequency,
                world_size=world_size,
                rank=rank,
                dist_address=trainer_config.dist_config.address,
                dist_port=trainer_config.dist_config.port,
                dist_backend=trainer_config.dist_config.backend,
                logger=logger,
                history=history
            )
    else:
        if is_ctc:
            trainer = CTCTrainer(
                optimizer=optimizer,
                criterion=criterion,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=trainer_config.epochs,
                log_steps_frequency=trainer_config.log_steps_frequency,
                device=trainer_config.device,
                logger=logger,
                history=history
            )
    trainer.fit()


def launch_asr_training(trainer_config, data_config, model_config):
    if trainer_config.dist_config is None:
        trainer = run_asr_trainer(
            0, 1, trainer_config, data_config, model_config
            )
        trainer.fit()
    else:
        world_size = trainer_config.dist_config.n_gpus
        spawn(
            run_asr_trainer,
            nprocs=trainer_config.dist_config.n_gpus,
            args=(
                world_size,
                trainer_config,
                data_config,
                model_config
                )
            )
