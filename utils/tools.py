import os
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, current_step=None, total_steps=None):
    if args.lradj == 'type3':
        if current_step is None or total_steps is None:
            return None
        if total_steps <= 0:
            raise ValueError('total_steps must be positive for type3 learning rate')

        current_step = min(max(current_step, 1), total_steps)
        warmup_steps = max(1, int(total_steps * 0.1))
        if current_step <= warmup_steps:
            lr = args.learning_rate * current_step / warmup_steps
        else:
            decay_steps = max(1, total_steps - warmup_steps)
            progress = (current_step - warmup_steps) / decay_steps
            min_lr_ratio = 0.1
            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = args.learning_rate * (
                min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio
            )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # type1/type2 are epoch-based and are updated by the epoch-end call.
    if current_step is not None:
        return None

    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        return None
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        return lr
    return None


class EarlyStopping:
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        mode='min',
        metric_name='Validation loss',
    ):
        if mode not in {'min', 'max'}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.metric_name = metric_name
        self.best_value = np.inf if mode == 'min' else -np.inf
        # Preserve the legacy public attribute for callers that inspect it.
        self.val_loss_min = np.inf

    def __call__(self, value, model, path):
        if not np.isfinite(value):
            raise ValueError(f'{self.metric_name} must be finite')
        improved = (
            value <= self.best_value - self.delta
            if self.mode == 'min'
            else value >= self.best_value + self.delta
        )
        if improved:
            self.save_checkpoint(value, model, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, value, model, path):
        if self.verbose:
            direction = 'decreased' if self.mode == 'min' else 'increased'
            print(
                f'{self.metric_name} {direction} '
                f'({self.best_value:.6f} --> {value:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.best_value = value
        self.val_loss_min = value


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
