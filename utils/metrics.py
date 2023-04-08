from typing import Tuple

import numpy as np


def MAE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred - true))


def MSE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean((pred - true) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    return np.sqrt(MSE(pred, true, mask_value))


def MAPE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))


def RSE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def MSPE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, mask=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param pred:
    :param true:
    :param mask
    :return:
    """
    assert type(pred) == type(true)
    mae = MAE(pred, true, mask)
    rmse = RMSE(pred, true, mask)
    mape = MAPE(pred, true, mask)
    mse = MSE(pred, true, mask)
    mspe = MSPE(pred, true, mask)

    return mae, mse, rmse, mape, mspe
