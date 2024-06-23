# visualize.py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from convertToPytoch import NextDayFireDataset
from ViT_model import VisionTransformer
from preprocess_data import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_inference(model, dataset, n_rows=5):
    model.eval()
    fig = plt.figure(figsize=(15, n_rows * 4))
    with torch.no_grad():
        for i in range(n_rows):
            inputs, labels = dataset[i]
            inputs = torch.tensor(inputs).unsqueeze(0).to(device)
            predictions = model(inputs).cpu().numpy()
            plt.subplot(n_rows, 3, i * 3 + 1)
            plt.title("Previous day fire")
            plt.imshow(inputs[0, -1, :, :], cmap='gray')
            plt.axis('off')
            plt.subplot(n_rows, 3, i * 3 + 2)
            plt.title("True next day fire")
            plt.imshow(labels[0, :, :], cmap='gray')
            plt.axis('off')
            plt.subplot(n_rows, 3, i * 3 + 3)
            plt.title("Predicted next day fire")
            plt.imshow(predictions[0, 0, :, :], cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.show()

file_pattern = 'path_to_your_tfrecord_files'
test_dataset = get_dataset(file_pattern, data_size=64, sample_size=32, batch_size=32, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=False, center_crop=True)
test_dataset = NextDayFireDataset(test_dataset)

model = VisionTransformer(input_shape, patch_size, embed_dim, num_heads, ff_dim, num_layers).to(device)
model.load_state_dict(torch.load('best_model.pth'))

show_inference(model, test_dataset, n_rows=5)
