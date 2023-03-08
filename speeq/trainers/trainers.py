"""This module contains different trainer classes, some of which utilize
distributed data parallelism (DDP), as well as a launch_training_job function.

Trainers:

- BaseTrainer: A basic trainer module.
- BaseDistTrainer: A basic distributed data parallel trainer module that is a subclass of BaseTrainer.
- CTCTrainer: A trainer module for CTC-based models that is a subclass of BaseTrainer.
- DistCTCTrainer: A trainer module for CTC models that utilizes distributed data parallelism, which is a subclass of both BaseDistTrainer and CTCTrainer.
- Seq2SeqTrainer: A trainer module for Seq2Seq models that is a subclass of BaseTrainer.
- DistSeq2SeqTrainer: A trainer module for Seq2Seq models that utilizes distributed data parallelism, which is a subclass of both BaseDistTrainer and Seq2SeqTrainer.
- TransducerTrainer: A trainer module for transducer-based models that is a subclass of BaseTrainer.
- DistTransducerTrainer: A trainer module for transducer models that utilizes distributed data parallelism, which is a subclass of both BaseDistTrainer and TransducerTrainer.


Function:

- launch_training_job: A function that launches a training job for a given configuration of trainer, data, and model objects. It takes in three arguments: trainer_config which is an object containing the configuration for the trainer, data_config which is an object containing the configuration for the data, and model_config which is an object containing the configuration for the model. The function returns None.
"""
import os
import time
from functools import partial
from math import inf
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce, barrier, init_process_group
from torch.multiprocessing import spawn
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from tqdm import tqdm

from speeq.config import ASRDataConfig, ModelConfig, TrainerConfig
from speeq.constants import HistoryKeys, LogCategories
from speeq.interfaces import IDataLoader, IScheduler, ITrainer
from speeq.utils.loggers import ILogger
from speeq.utils.utils import get_key_tag, has_bnorm

from .decorators import export_ckpt, step_log


class BaseTrainer(ITrainer):
    """Builds the basic trainer module

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
        criterion: Module,
        model: Module,
        train_loader: IDataLoader,
        test_loader: IDataLoader,
        epochs: int,
        log_steps_frequency: int,
        logger: ILogger,
        outdir: Union[str, Path],
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
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
        self.grad_acc_steps = grad_acc_steps
        if HistoryKeys.min_loss.value not in self.history:
            self.history[HistoryKeys.min_loss.value] = inf

    def backward_pass(self, loss: Tensor) -> None:
        """This method performs a backward pass on the model parameters to update
        them based on the provided loss tensor.

        Args:
            loss (Tensor): The loss tensor.
        """
        loss = loss / self.grad_acc_steps
        loss.backward()
        if self.grad_clip_thresh is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_thresh,
                norm_type=self.grad_clip_norm_type,
            )
        if self.counter % self.grad_acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def fit(self):
        """Fits the model on the training data."""
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

    @step_log(key=HistoryKeys.train_loss.value, category=LogCategories.batches.value)
    def train_step(self, batch: Tuple[Tensor]) -> float:
        """This method represents a single step in the training process. It
        performs a forward pass, calculates the loss, and then performs a
        backward pass to update the model parameters.

            Args:

            batch (Tuple[Tensor]): The input batch to be processed.
            Returns:

            float: The loss value for this step.
        """
        loss = self.forward_pass(batch)
        self.backward_pass(loss)
        return loss.item()

    @step_log(key=HistoryKeys.train_loss.value, category=LogCategories.epochs.value)
    def train(self) -> float:
        """The main training loop, where the function iterate over the training
        examples and perform forward and backward pass.

        Returns:

            float: The average loss over all training examples.

        """
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
                    value=total_loss / (i + 1),
                )
            self.counter += 1
        return total_loss / len(self.train_loader)

    @export_ckpt(key=HistoryKeys.test_loss.value, category=LogCategories.steps.value)
    @step_log(key=HistoryKeys.test_loss.value, category=LogCategories.steps.value)
    @torch.no_grad()
    def test(self) -> float:
        """Performing a model test on the testing data

        Returns:
            float: The average test loss.
        """
        self.model.eval()
        total_loss = 0.0
        for batch in self.test_loader:
            loss = self.forward_pass(batch)
            total_loss += loss.item()
        total_loss /= len(self.test_loader)
        return total_loss

    @property
    def is_master(self):
        return True


class BaseDistTrainer(BaseTrainer):
    """Builds the basic distributed data parallel trainer module

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        rank (int): The process index.

        world_size (int): The number of nodes/processes.

        dist_address (str): The address of the master node.

        dist_port (int): The port of the master node.

        dist_backend (str): The backend used for DDP.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
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
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history={},
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )
        self.rank = rank
        self.world_size = world_size
        self.dist_port = dist_port
        self.dist_address = dist_address
        self.dist_backend = dist_backend
        self.init_dist()
        self.has_bnorm = has_bnorm(self.model)
        self.model.to(f"cuda:{rank}")
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(self.model, device_ids=[self.rank])

    def init_dist(self):
        """initialize the distributed training process"""
        os.environ["MASTER_ADDR"] = self.dist_address
        os.environ["MASTER_PORT"] = str(self.dist_port)
        init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_address,
            world_size=self.world_size,
            rank=self.rank,
        )

    @property
    def is_master(self):
        return self.rank == 0

    def _all_reduce_loss(self, total_loss: float, counter: int) -> Tensor:
        total = torch.tensor([total_loss / counter]).cuda(self.rank)
        all_reduce(total, op=ReduceOp.SUM)
        return total / self.world_size

    def backward_pass(self, loss: Tensor) -> None:
        """This method performs a backward pass on the model parameters to update
        them based on the provided loss tensor.

        Args:
            loss (Tensor): The loss tensor.
        """
        loss = loss / self.grad_acc_steps
        loss.backward()
        if self.grad_clip_thresh is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_thresh,
                norm_type=self.grad_clip_norm_type,
            )
        if self.counter % self.grad_acc_steps == 0:
            for param in self.model.parameters():
                param.grad.data /= self.world_size
            self.optimizer.step()
            self.optimizer.zero_grad()

    @step_log(key=HistoryKeys.train_loss.value, category=LogCategories.epochs.value)
    def train(self) -> float:
        """The main training loop that run on one of the processes, where the
        function iterate over the training examples and perform forward and backward pass.

        Returns:

            float: The average loss over all training examples from all processes.

        """
        self.model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(self.train_loader)):
            loss = self.train_step(batch)
            total_loss += loss
            if self.counter % self.log_steps_frequency == 0:
                total = self._all_reduce_loss(total_loss, i + 1)
                if self.is_master or self.has_bnorm is True:
                    """The extra condition to solve a dummy issue caused when
                    we have DDP with batch norm, it works only if the
                    evaluation is done on all nodes!, the link below
                    is similar issue
                    discuss.pytorch.org/t/validation-hangs-up-when-using-ddp-and-syncbatchnorm/104831
                    """
                    self.inline_log(
                        key=HistoryKeys.train_loss.value,
                        category=LogCategories.steps.value,
                        value=total.item(),
                    )
                    self.test()
                    self.model.train()
                if self.has_bnorm is False:
                    barrier()
            self.counter += 1
        return self._all_reduce_loss(total_loss, len(self.train_loader)).item()

    def fit(self):
        """Fits the model on the training data, and logs the results on the master
        node only.
        """
        for _ in range(self.epochs):
            self.train()
            if self.is_master or self.has_bnorm is True:
                """The extra condition to solve a dummy issue caused when
                we have DDP with batch norm, it works only if the evaluation is done on all nodes!
                the link below is similar issue
                discuss.pytorch.org/t/validation-hangs-up-when-using-ddp-and-syncbatchnorm/104831
                """
                self.logger.log(self.history)
            barrier()


class CTCTrainer(BaseTrainer):
    """A trainer module for CTC-based models.

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
        criterion: Module,
        model: Module,
        train_loader: IDataLoader,
        test_loader: IDataLoader,
        epochs: int,
        log_steps_frequency: int,
        device: str,
        logger: ILogger,
        outdir: Union[str, Path],
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )
        self.device = device
        self.model.to(device)

    def forward_pass(self, batch: Tuple[Tensor]) -> Tensor:
        batch = [item.to(self.device) for item in batch]
        [speech, speech_mask, text, text_mask] = batch
        preds, lengths = self.model(speech, speech_mask)
        # preds of shape [T, B, C]
        loss = self.criterion(preds, text, lengths, text_mask.sum(dim=-1))
        return loss


class DistCTCTrainer(BaseDistTrainer, CTCTrainer):
    """A trainer module for CTC models that utilizes distributed data parallelism.

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        rank (int): The process index.

        world_size (int): The number of nodes/processes.

        dist_address (str): The address of the master node.

        dist_port (int): The port of the master node.

        dist_backend (str): The backend used for DDP.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
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
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
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
            device=f"cuda:{rank}",
            logger=logger,
            outdir=outdir,
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )


class Seq2SeqTrainer(BaseTrainer):
    """A trainer module for Seq2Seq models.

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
        criterion: Module,
        model: Module,
        train_loader: IDataLoader,
        test_loader: IDataLoader,
        epochs: int,
        log_steps_frequency: int,
        device: str,
        logger: ILogger,
        outdir: Union[str, Path],
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )
        self.device = device
        self.model.to(device)

    def forward_pass(self, batch: Tuple[Tensor]) -> Tensor:
        batch = [item.to(self.device) for item in batch]
        [speech, speech_mask, text, text_mask] = batch
        preds = self.model(speech, speech_mask, text, text_mask)
        loss = self.criterion(preds, text, text_mask)
        return loss


class DistSeq2SeqTrainer(BaseDistTrainer, Seq2SeqTrainer):
    """A trainer module for Seq2Seq models that utilizes distributed data parallelism.

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        rank (int): The process index.

        world_size (int): The number of nodes/processes.

        dist_address (str): The address of the master node.

        dist_port (int): The port of the master node.

        dist_backend (str): The backend used for DDP.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
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
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
    ) -> None:
        Seq2SeqTrainer.__init__(
            self,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            device=f"cuda:{rank}",
            logger=logger,
            outdir=outdir,
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )


class TransducerTrainer(BaseTrainer):
    """A trainer module for transducer-based models.

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
        criterion: Module,
        model: Module,
        train_loader: IDataLoader,
        test_loader: IDataLoader,
        epochs: int,
        log_steps_frequency: int,
        device: str,
        logger: ILogger,
        outdir: Union[str, Path],
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )
        self.device = device
        self.model.to(device)

    def forward_pass(self, batch: Tuple[Tensor]) -> Tensor:
        """This method conducts a forward pass on the CTC model.

        Args:

            batch (Tuple[Tensor]): The input batch containing the speech, speech
            length, text, and text length tensors, in that order.

        Returns:

            Tensor: A tensor representing the loss.
        """
        batch = [item.to(self.device) for item in batch]
        [speech, speech_mask, text, text_mask] = batch
        preds, speech_len, text_len = self.model(speech, speech_mask, text, text_mask)
        text, speech_len, text_len = (text.int(), speech_len.int(), text_len.int())
        loss = self.criterion(preds, speech_len, text, text_len)
        return loss


class DistTransducerTrainer(BaseDistTrainer, TransducerTrainer):
    """A trainer module for transducer models that utilizes distributed data parallelism.

    Args:
        optimizer (Union[Optimizer, IScheduler]): The optimizer or the wrapped
        optimizer that will be used during the training.

        criterion (Module): The loss fucntion that will be used
        during the training process.

        model (Module): The model.

        train_loader (ILoader): The loader for the training data.

        test_loader (ILoader): The loader for the testing data.

        epochs (int): The number of epochs.

        log_steps_frequency (int): The frequency at which to log results.

        logger (ILogger): The logger to be used.

        outdir (Union[str, Path]):  The directory to save checkpoints.

        rank (int): The process index.

        world_size (int): The number of nodes/processes.

        dist_address (str): The address of the master node.

        dist_port (int): The port of the master node.

        dist_backend (str): The backend used for DDP.

        grad_acc_steps (int): The number of steps to accumulate gradients
        over. Default 1.

        grad_clip_thresh (Union[None, float]): The maximum norm of the gradients.
        Default None.

        grad_clip_norm_type (float): The type of p-norm used. Default 2.0.

        history (dict): The training history, if available. Default {}.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, IScheduler],
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
        grad_acc_steps: int = 1,
        grad_clip_thresh: Union[None, float] = None,
        grad_clip_norm_type: float = 2.0,
        history: dict = {},
    ) -> None:
        TransducerTrainer.__init__(
            self,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            log_steps_frequency=log_steps_frequency,
            device=f"cuda:{rank}",
            logger=logger,
            outdir=outdir,
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
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
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            grad_clip_norm_type=grad_clip_norm_type,
            history=history,
        )


def _run_trainer(
    rank: int,
    world_size: int,
    trainer_config: TrainerConfig,
    data_config: ASRDataConfig,
    model_config: ModelConfig,
) -> None:
    if rank != 0:
        # To make sure the master node created any dependancies
        # This can be replaced if we pass the rank to the
        # factories depend on the master node
        time.sleep(5)
    from speeq.trainers.registry import get_asr_trainer

    trainer = get_asr_trainer(
        rank=rank,
        world_size=world_size,
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config,
    )
    trainer.fit()


def launch_training_job(
    trainer_config: object, data_config: object, model_config: object
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
        model_config=model_config,
    )
    if trainer_config.dist_config is None:
        trainer_launcher(rank=0, world_size=1)
    else:
        world_size = trainer_config.dist_config.n_gpus
        spawn(
            trainer_launcher,
            nprocs=trainer_config.dist_config.n_gpus,
            args=(world_size,),
        )
