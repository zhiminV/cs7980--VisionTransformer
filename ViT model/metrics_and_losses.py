# metrics_and_losses.py
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def IoU_metric(real_mask, predicted_mask):
    real_mask = (real_mask > 0.5).float()
    predicted_mask = (predicted_mask > 0.5).float()
    intersection = (real_mask * predicted_mask).sum((1, 2))
    union = real_mask.sum((1, 2)) + predicted_mask.sum((1, 2)) - intersection
    IoU = (intersection + 1e-6) / (union + 1e-6)
    return IoU.mean().item()

def recall_metric(real_mask, predicted_mask):
    real_mask = (real_mask > 0.5).float()
    predicted_mask = (predicted_mask > 0.5).float()
    true_positives = (real_mask * predicted_mask).sum((1, 2))
    actual_positives = real_mask.sum((1, 2))
    recall = (true_positives + 1e-6) / (actual_positives + 1e-6)
    return recall.mean().item()

def precision_metric(real_mask, predicted_mask):
    real_mask = (real_mask > 0.5).float()
    predicted_mask = (predicted_mask > 0.5).float()
    true_positives = (real_mask * predicted_mask).sum((1, 2))
    predicted_positives = predicted_mask.sum((1, 2))
    precision = (true_positives + 1e-6) / (predicted_positives + 1e-6)
    return precision.mean().item()

def bce_dice_loss(y_true, y_pred):
    bce = F.binary_cross_entropy(y_pred, y_true)
    smooth = 1e-6
    intersection = (y_true * y_pred).sum()
    dice = 1 - (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    return bce + dice
