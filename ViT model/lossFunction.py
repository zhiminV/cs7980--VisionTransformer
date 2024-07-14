import torch
import torch.nn.functional as F

BATCH_SIZE = 32  # Ensure this matches your batch size

def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, smooth=1e-6) -> torch.Tensor:
    y_true_f = y_true.view(BATCH_SIZE, -1)
    y_pred_f = y_pred.view(BATCH_SIZE, -1)
    intersection = (y_true_f * y_pred_f).sum(1)
    return 1 - (2. * intersection + smooth) / (y_true_f.sum(1) + y_pred_f.sum(1) + smooth)

def weighted_bincrossentropy(y_true: torch.Tensor, y_pred: torch.Tensor, weight_zero: float = 0.01, weight_one: float = 1) -> torch.Tensor:
    bin_crossentropy = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    weights = y_true * weight_one + (1. - y_true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy
    return weighted_bin_crossentropy.mean()

def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return weighted_bincrossentropy(y_true, y_pred) + dice_coef(y_true, y_pred)
