import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from vit_model import VIT
from preprocess_data import get_dataset, tf_dataset_to_numpy

BATCH_SIZE = 32

train_dataset_tf = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord', 
                               data_size=64, sample_size=32, batch_size=BATCH_SIZE, 
                               num_in_channels=12, compression_type=None, 
                               clip_and_normalize=True, clip_and_rescale=False, 
                               random_crop=True, center_crop=False)

validation_dataset_tf = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/val/*_ongoing_*.tfrecord', 
                                    data_size=64, sample_size=32, batch_size=BATCH_SIZE, 
                                    num_in_channels=12, compression_type=None, 
                                    clip_and_normalize=True, clip_and_rescale=False, 
                                    random_crop=True, center_crop=False)

test_dataset_tf = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/test/*_ongoing_*.tfrecord', 
                              data_size=64, sample_size=32, batch_size=BATCH_SIZE, 
                              num_in_channels=12, compression_type=None, 
                              clip_and_normalize=True, clip_and_rescale=False, 
                              random_crop=True, center_crop=False)

train_data = tf_dataset_to_numpy(train_dataset_tf)
val_data = tf_dataset_to_numpy(validation_dataset_tf)
test_data = tf_dataset_to_numpy(test_dataset_tf)

class FireDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_img, output_img = self.data[idx]
        
        # Check dimensions before permuting
        if input_img.ndim == 3:
            input_img = torch.tensor(input_img, dtype=torch.float32).permute(2, 0, 1)
        elif input_img.ndim == 4:
            input_img = torch.tensor(input_img, dtype=torch.float32).squeeze().permute(2, 0, 1)
        
        if output_img.ndim == 3:
            output_img = torch.tensor(output_img, dtype=torch.float32).permute(2, 0, 1)
        elif output_img.ndim == 4:
            output_img = torch.tensor(output_img, dtype=torch.float32).squeeze().permute(2, 0, 1)

        return input_img, output_img

train_dataset = FireDataset(train_data)
validation_dataset = FireDataset(val_data)
test_dataset = FireDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)

# Configuration for ViT
config = {
    'image_height': 64,
    'image_width': 64,
    'im_channels': 12,
    'patch_height': 16,
    'patch_width': 16,
    'emb_dim': 256,
    'patch_emb_drop': 0.1,
    'n_heads': 8,
    'head_dim': 32,
    'ff_dim': 1024,
    'ff_drop': 0.1,
    'n_layers': 6,
    'num_classes': 1,
    'dropout': 0.1
}

# Initialize the model, criterion, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VIT(config).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), "vit_model.pth")
