import torch
import torch.nn as nn
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
        
        if predictions.shape != labels.shape:
            predictions = torch.tensor(predictions, dtype=torch.float32)
            predictions = nn.functional.interpolate(predictions, size=labels.shape[2:], mode='bilinear', align_corners=False).to(device)
        
        # Clip target values to the range [0, 1]
        labels = torch.clamp(labels, 0, 1)
        
        # Move predictions and labels to CPU before converting to numpy
        predictions_cpu = predictions.cpu().numpy()
        labels_cpu = labels.cpu().numpy()

        for i in range(inputs.shape[0]):
            IoU_measures.append(IoU_metric(labels_cpu[i, 0], predictions_cpu[i]))
            recall_measures.append(recall_metric(labels_cpu[i, 0], predictions_cpu[i]))
            precision_measures.append(precision_metric(labels_cpu[i, 0], predictions_cpu[i]))
        
        # Calculate and append the loss
        loss = bce_dice_loss(labels, torch.tensor(predictions, dtype=torch.float32).to(device))
        losses.append(loss.item())  # Ensure the loss is a scalar and convert it to a Python float

    mean_IoU = np.mean(IoU_measures)
    mean_recall = np.mean(recall_measures)
    mean_precision = np.mean(precision_measures)
    mean_loss = np.mean(losses)
    return mean_IoU, mean_recall, mean_precision, mean_loss
