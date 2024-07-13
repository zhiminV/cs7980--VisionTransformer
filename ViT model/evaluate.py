import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from metrics import IoU_metric, recall_metric, precision_metric
from lossFunction import bce_dice_loss

def evaluate_model(prediction_function, eval_dataset, device='cuda'):
    IoU_measures = []
    recall_measures = []
    precision_measures = []
    losses = []
    
    for inputs, labels in tqdm(eval_dataset):
        inputs = torch.from_numpy(inputs.numpy()).permute(0, 3, 1, 2).float().to(device)
        labels = torch.from_numpy(labels.numpy()).permute(0, 3, 1, 2).float().to(device)
        
        predictions = prediction_function(inputs)
        
        # Convert predictions and labels to TensorFlow tensors for metric computation
        predictions_tf = tf.convert_to_tensor(predictions)
        labels_tf = tf.convert_to_tensor(labels.detach().cpu().numpy())
        
        for i in range(inputs.shape[0]):
            IoU_measures.append(IoU_metric(labels_tf[i, :, :, 0], predictions_tf[i, :, :]))
            recall_measures.append(recall_metric(labels_tf[i, :, :, 0], predictions_tf[i, :, :]))
            precision_measures.append(precision_metric(labels_tf[i, :, :, 0], predictions_tf[i, :, :]))
        
        labels_cleared = tf.where(labels_tf < 0, 0, labels_tf)
        losses.append(bce_dice_loss(labels_cleared, tf.expand_dims(predictions_tf, axis=-1)))
            
    mean_IoU = np.mean(IoU_measures)
    mean_recall = np.mean(recall_measures)
    mean_precision = np.mean(precision_measures)
    mean_loss = np.mean(losses)
    return mean_IoU, mean_recall, mean_precision, mean_loss
