import random

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from speeq.trainers import criterions


class TestCTCLoss:
    @pytest.mark.parametrize(
        (
            "batch_size",
            "speech_len",
            "n_classes",
            "text_len",
            "speech_pad_lens",
            "text_pad_lens",
        ),
        (
            (3, 10, 5, 7, [0, 2, 4], [4, 0, 5]),
            (3, 15, 3, 8, [0, 2, 4], [4, 0, 5]),
            (3, 12, 4, 9, [0, 2, 4], [4, 0, 5]),
        ),
    )
    def test_forward(
        self,
        batcher,
        batch_size,
        speech_len,
        n_classes,
        text_len,
        speech_pad_lens,
        text_pad_lens,
    ):
        feat_size = 10
        model = nn.Linear(10, n_classes)
        preds = model(batcher(speech_len, batch_size, feat_size))
        preds = F.log_softmax(preds, dim=-1)
        pred_lens = torch.LongTensor([speech_len - item for item in speech_pad_lens])
        text_lens = torch.LongTensor([text_len - item for item in text_pad_lens])
        target = []
        for pad_len in text_pad_lens:
            r = [random.randint(1, n_classes) for _ in range(text_len - pad_len)]
            r = r + [0] * pad_len
            target.append(r)
        target = torch.LongTensor(target)
        criterion = criterions.CTCLoss(blank_id=0)
        loss = criterion(preds, target, pred_lens, text_lens)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestCrossEntropyLoss:
    @pytest.mark.parametrize(
        ("batch_size", "n_classes", "text_len", "text_pad_lens"),
        (
            (3, 5, 7, [4, 0, 5]),
            (3, 3, 8, [4, 0, 5]),
            (3, 4, 9, [4, 0, 5]),
        ),
    )
    def test_forward(self, batcher, batch_size, n_classes, text_len, text_pad_lens):
        feat_size = 10
        model = nn.Linear(10, n_classes)
        preds = model(batcher(batch_size, text_len, feat_size))
        preds = F.softmax(preds, dim=-1)
        target = []
        for pad_len in text_pad_lens:
            r = [random.randint(1, n_classes - 1) for _ in range(text_len - pad_len)]
            r = r + [0] * pad_len
            target.append(r)
        target = torch.LongTensor(target)
        criterion = criterions.CrossEntropyLoss(pad_id=0)
        loss = criterion(preds, target)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestNLLLoss:
    @pytest.mark.parametrize(
        ("batch_size", "n_classes", "text_len", "text_pad_lens"),
        (
            (3, 5, 7, [4, 0, 5]),
            (3, 3, 8, [4, 0, 5]),
            (3, 4, 9, [4, 0, 5]),
        ),
    )
    def test_forward(self, batcher, batch_size, n_classes, text_len, text_pad_lens):
        feat_size = 10
        model = nn.Linear(10, n_classes)
        preds = model(batcher(batch_size, text_len, feat_size))
        preds = F.log_softmax(preds, dim=-1)
        target = []
        for pad_len in text_pad_lens:
            r = [random.randint(1, n_classes - 1) for _ in range(text_len - pad_len)]
            r = r + [0] * pad_len
            target.append(r)
        target = torch.LongTensor(target)
        criterion = criterions.NLLLoss(pad_id=0)
        loss = criterion(preds, target)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestRNNTLoss:
    @pytest.mark.parametrize(
        (
            "batch_size",
            "speech_len",
            "n_classes",
            "text_len",
            "speech_pad_lens",
            "text_pad_lens",
        ),
        (
            (3, 10, 5, 7, [0, 2, 4], [4, 0, 5]),
            (3, 15, 3, 8, [0, 2, 4], [4, 0, 5]),
            (3, 12, 4, 9, [0, 2, 4], [4, 0, 5]),
        ),
    )
    def test_forward(
        self,
        batcher,
        batch_size,
        speech_len,
        n_classes,
        text_len,
        speech_pad_lens,
        text_pad_lens,
    ):
        feat_size = 10
        model = nn.Linear(10, n_classes)
        preds = model(batcher(batch_size, speech_len * text_len, feat_size))
        preds = preds.view(batch_size, speech_len, text_len, n_classes)
        pred_lens = torch.IntTensor([speech_len - item for item in speech_pad_lens])
        text_lens = torch.IntTensor([text_len - item for item in text_pad_lens])
        target = []
        for pad_len in text_pad_lens:
            r = [random.randint(1, n_classes) for _ in range(text_len - pad_len)]
            r = r + [0] * pad_len
            target.append(r)
        target = torch.IntTensor(target)
        criterion = criterions.RNNTLoss(blank_id=0)
        loss = criterion(preds, pred_lens, target, text_lens)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None
