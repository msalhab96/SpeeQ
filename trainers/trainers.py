from functools import partial
from math import inf
import os
from pathlib import Path
import time
import torch
from torch import Tensor
from typing import Tuple, Union
from trainers.decorators import export_ckpt, step_log
from utils.loggers import ILogger
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

from utils.utils import get_key_tag


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
        outdir (Union[str, Path]): The output directory to save
        checkpoints into.
        grad_clip_thresh (Union[None, float]): max norm of the gradients.
        Default None.
        grad_clip_norm_type (float): type of the used p-norm. Default 2.0.
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
            outdir: Union[str, Path],
            grad_clip_thresh: Union[None, float] = None,
            grad_clip_norm_type: float = 2.0,
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
        self.outdir = outdir
        self.grad_clip_thresh = grad_clip_thresh
        self.grad_clip_norm_type = grad_clip_norm_type
        self.history = history
        self.counter = 1
        self.min_loss = inf

    def backward_pass(self, loss: Tensor) -> None:
        if self.grad_clip_thresh is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model,
                max_norm=self.grad_clip_thresh,
                norm_type=self.grad_clip_norm_type
                )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit(self):
        for _ in range(self.epochs):
            self.train()
            self.logger.log(self.history)

    def inline_log(self, key: str, category: str, value: int):
        tag = get_key_tag(key=key, category=category)
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
        outdir (Union[str, Path]): The output directory to save
        checkpoints into.
        rank (int): The process index.
        world_size (int): The number of nodes/processes.
        dist_address (str): The address of the master node.
        dist_port (int): The port of the master node.
        dist_backend (str): The backend used for DDP.
        grad_clip_thresh (Union[None, float]): max norm of the gradients.
        Default None.
        grad_clip_norm_type (float): type of the used p-norm. Default 2.0.
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
            outdir: Union[str, Path],
            rank: int,
            world_size: int,
            dist_address: str,
            dist_port: int,
            dist_backend: str,
            grad_clip_thresh: Union[None, float] = None,
            grad_clip_norm_type: float = 2.0,
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
            outdir=outdir,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history
        )
        self.rank = rank
        self.world_size = world_size
        self.dist_port = dist_port
        self.dist_address = dist_address
        self.dist_backend = dist_backend
        self.init_dist()
        self.has_bnorm = self.model.has_bnorm
        self.model.to(f'cuda:{rank}')
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.model
        )
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
            outdir: Union[str, Path],
            grad_clip_thresh: Union[None, float] = None,
            grad_clip_norm_type: float = 2.0,
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
            outdir=outdir,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history
        )
        self.device = device
        self.model.to(device)

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
            if self.counter % self.log_steps_frequency == 0:
                self.test()
                self.model.train()
                self.inline_log(
                    key=HistoryKeys.train_loss.value,
                    category=LogCategories.steps.value,
                    value=total_loss / (i + 1)
                    )
            self.counter += 1
        return total_loss / len(self.train_loader)

    @export_ckpt(
        key=HistoryKeys.test_loss.value,
        category=LogCategories.steps.value
        )
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
        total_loss /= len(self.test_loader)
        return total_loss


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
            outdir: Union[str, Path],
            log_steps_frequency: int,
            rank: int,
            world_size: int,
            dist_address: int,
            dist_port: int,
            dist_backend: str,
            grad_clip_thresh: Union[None, float] = None,
            grad_clip_norm_type: float = 2.0,
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
            outdir=outdir,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
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
            outdir=outdir,
            rank=rank,
            world_size=world_size,
            dist_address=dist_address,
            dist_port=dist_port,
            dist_backend=dist_backend,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
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
            if self.counter % self.log_steps_frequency == 0:
                total = self._all_reduce_loss(total_loss, i + 1)
                if self.is_master:
                    self.inline_log(
                        key=HistoryKeys.train_loss.value,
                        category=LogCategories.steps.value,
                        value=total.item()
                        )
                    self.test()
                    self.model.train()
                if self.has_bnorm is False:
                    barrier()
            self.counter += 1
        return self._all_reduce_loss(total_loss, len(self.train_loader)).item()

    def fit(self):
        for _ in range(self.epochs):
            self.train()
            if self.is_master:
                self.logger.log(self.history)
            barrier()


def _run_trainer(
        rank: int,
        world_size: int,
        trainer_config,
        data_config,
        model_config
        ) -> None:
    if rank != 0:
        # To make sure the master node created any dependancies
        # This can be replaced if we pass the rank to the
        # factories depend on the master node
        time.sleep(5)
    from trainers.registry import get_asr_trainer
    trainer = get_asr_trainer(
        rank=rank, world_size=world_size,
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config
    )
    trainer.fit()


def launch_training_job(
        trainer_config: object,
        data_config: object,
        model_config: object
        ) -> None:
    """Launches ASR training job by constructing
    a trainer from the passed configuration and run it
    on single or multiple GPUS.

    Args:
        trainer_config (object): Trainer configuration object.
        data_config (object): Data configuration object.
        model_config (object): Model configuration object.
    """
    trainer_launcher = partial(
        _run_trainer,
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config
        )
    if trainer_config.dist_config is None:
        trainer_launcher(
            rank=0,
            world_size=1
            )
    else:
        world_size = trainer_config.dist_config.n_gpus
        spawn(
            trainer_launcher,
            nprocs=trainer_config.dist_config.n_gpus,
            args=(world_size,)
            )
