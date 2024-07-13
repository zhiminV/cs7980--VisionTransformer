import tensorflow as tf
import numpy as np
from tqdm import tqdm
from metrics import IoU_metric, recall_metric, precision_metric
from lossFunction import bce_dice_loss

def evaluate_model(prediction_function, eval_dataset):
    IoU_measures = []
    recall_measures = []
    precision_measures = []
    losses = []
    
    for inputs, labels in tqdm(eval_dataset):
        predictions = prediction_function(inputs)
        for i in range(inputs.shape[0]):
            IoU_measures.append(IoU_metric(labels[i, :, :,  0], predictions[i, :, :]))
            recall_measures.append(recall_metric(labels[i, :, :,  0], predictions[i, :, :]))
            precision_measures.append(precision_metric(labels[i, :, :,  0], predictions[i, :, :]))
        labels_cleared = tf.where(labels < 0, 0, labels)
        losses.append(bce_dice_loss(labels_cleared, tf.expand_dims(tf.cast(predictions, tf.float32), axis=-1)))
            
    mean_IoU = np.mean(IoU_measures)
    mean_recall = np.mean(recall_measures)
    mean_precision = np.mean(precision_measures)
    mean_loss = np.mean(losses)
    return mean_IoU, mean_recall, mean_precision, mean_loss
