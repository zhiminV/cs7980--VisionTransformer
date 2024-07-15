import torch
import torch.nn.functional as F

def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, smooth=1e-6) -> torch.Tensor:
    y_true_f = y_true.view(y_true.size(0), -1)
    y_pred_f = y_pred.view(y_true.size(0), -1)
    intersection = (y_true_f * y_pred_f).sum(1)
    dice = (2. * intersection + smooth) / (y_true_f.sum(1) + y_pred_f.sum(1) + smooth)
    return 1 - dice.mean()

def weighted_bincrossentropy(y_true: torch.Tensor, y_pred: torch.Tensor, weight_zero: float = 0.01, weight_one: float = 1) -> torch.Tensor:
    bin_crossentropy = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    weights = y_true * weight_one + (1. - y_true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy
    return weighted_bin_crossentropy.mean()

def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    bce_loss = weighted_bincrossentropy(y_true, y_pred)
    dice_loss = dice_coef(y_true, y_pred)
    return bce_loss + dice_loss
