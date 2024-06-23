# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from convertToPytoch import NextDayFireDataset
from ViT_model import VisionTransformer
from preprocess_data import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, eval_loader, criterion):
    model.eval()
    eval_loss = 0.0
    IoU_measures = []
    recall_measures = []
    precision_measures = []
    with torch.no_grad():
        for inputs, masks in eval_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            eval_loss += loss.item() * inputs.size(0)
            # Add your IoU, precision, recall calculation here
    eval_loss /= len(eval_loader.dataset)
    return eval_loss, IoU_measures, recall_measures, precision_measures

file_pattern = 'path_to_your_tfrecord_files'
eval_dataset = get_dataset(file_pattern, data_size=64, sample_size=32, batch_size=32, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=False, center_crop=True)
eval_dataset = NextDayFireDataset(eval_dataset)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = VisionTransformer(input_shape, patch_size, embed_dim, num_heads, ff_dim, num_layers).to(device)
model.load_state_dict(torch.load('best_model.pth'))

criterion = nn.BCELoss()
eval_loss, IoU_measures, recall_measures, precision_measures = evaluate_model(model, eval_loader, criterion)
print(f'Eval Loss: {eval_loss:.4f}')
