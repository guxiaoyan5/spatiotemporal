import logging
import os
import time
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from utils.logger import get_logger
from utils.metrics import metric


class BaseTrainer:
    def __init__(self, model: nn.Module, loss_fn, data, base_lr: float, steps, lr_decay_ratio,
                 log_dir: str, n_exp: int, save_iter: int = 300, clip_grad_value: Optional[float] = None,
                 max_epochs: Optional[int] = 1000, patience: Optional[int] = 1000,
                 device: Optional[Union[torch.device, str]] = None, **args):
        super().__init__()

        self._logger = get_logger(log_dir, __name__, 'info_{}.log'.format(n_exp), level=logging.INFO)
        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)

        self._model = model
        self._model.to(self._device)
        self._logger.info("the number of parameters: {}".format(self.model.param_num()))

        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)
        self._loss_fn = loss_fn
        self._lr_decay_ratio = lr_decay_ratio
        self._steps = steps
        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer, steps, gamma=lr_decay_ratio)
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._data = data

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'final_model_{}.pt'.format(n_exp))
        torch.save(self.model.state_dict(), filename)
        return filename

    def load_model(self, save_path, n_exp):
        filename = 'final_model_{}.pt'.format(n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_loss):
        self.logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch, best_loss))
        np.savetxt(os.path.join(self.save_path, 'val_loss_{}.txt'.format(self._n_exp)), best_loss, fmt='%.4f',
                   delimiter=',')

    def train_batch(self, *batch):
        label = batch[-1].to(self.device)
        inputs = (i.to(self.device) for i in batch[:-1])
        self.optimizer.zero_grad()
        pred = self.model(*inputs)
        loss = self.loss_fn(pred, label)
        loss.backward()
        if self._clip_grad_value is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, (*batch,) in enumerate(self.data['train_loader']):
                train_losses.append(self.train_batch(*batch))
                iter += 1
                if iter is not None:
                    if iter % self._save_iter == 0:  # iteration needs to be checked
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                                                                                  self._max_epochs,
                                                                                                  iter,
                                                                                                  np.mean(train_losses),
                                                                                                  val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch

            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                      '{:.1f}s'.format(epoch, self._max_epochs, iter, np.mean(train_losses), val_loss, new_lr,
                                       (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):  # error saving criterion
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (*batch,) in enumerate(self.data['val_loader']):
                label = batch[-1].to(self.device)
                inputs = (i.to(self.device) for i in batch[:-1])
                pred = self.model(*inputs)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mae = self.loss_fn(preds, labels).item()
        return mae

    def test_batch(self, *batch):
        '''
        the test process of a batch
        '''
        label = batch[-1]
        inputs = batch[:-1]
        pred = self.model(*inputs)
        return pred, label

    def test(self, mode='test'):
        self.load_model(self.save_path, self._n_exp)
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (*batch,) in enumerate(self.data[mode + '_loader']):
                pred, label = self.test_batch(*batch)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0).detach().numpy() * self.data["pm25_std"] + self.data["pm25_mean"]
        preds = torch.cat(preds, dim=0).detach().numpy() * self.data["pm25_std"] + self.data["pm25_mean"]

        horizon = labels.shape[1]
        for i in range(horizon):
            mae, mse, rmse, mape, mspe = metric(preds[:, i, ...], labels[:, i, ...])
            self._logger.info(
                'horizon {}: mae {:.4f}, mse {:.4f}, rmse {:.4f}, mape {:.4f}, mspe {:.4f}'.format(i, mae, mse, rmse,
                                                                                                   mape, mspe))
        mae, mse, rmse, mape, mspe = metric(preds, labels)
        self._logger.info(
            'mae {:.4f}, mse {:.4f}, rmse {:.4f}, mape {:.4f}, mspe {:.4f}'.format(mae, mse, rmse, mape, mspe))
        np.save(os.path.join(self._save_path, mode + "_pred.npy"), preds)
        np.save(os.path.join(self._save_path, mode + "_true.npy"), labels)
