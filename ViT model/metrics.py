import tensorflow as tf
import numpy as np

def IoU_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    intersection = np.logical_and(real_mask, predicted_mask)
    union = np.logical_or(real_mask, predicted_mask)
    if np.sum(union) == 0:
        return 1
    return np.sum(intersection) / np.sum(union)

def recall_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    true_positives = np.sum(np.logical_and(real_mask, predicted_mask))
    actual_positives = np.sum(real_mask)
    if actual_positives == 0:
        return 1
    return true_positives / actual_positives

def precision_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    true_positives = np.sum(np.logical_and(real_mask, predicted_mask))
    predicted_positives = np.sum(predicted_mask)
    if predicted_positives == 0:
        return 1
    return true_positives / predicted_positives
