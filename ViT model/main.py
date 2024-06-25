import matplotlib.pyplot as plt
from preprocess_data import NextDayFireDataset
from patchEmbedding import PatchEmbedding
# Constants
DATA_SIZE = 64
PATCH_SIZE = 64  # Can change as needed; in Bronte's paper, she made it 32
BATCH_SIZE = 32

# Load dataset and print input image size and channels
train_dataset = NextDayFireDataset(
    file_pattern='/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord',
    data_size=DATA_SIZE,
    sample_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_in_channels=12,
    compression_type=None,
    clip_and_normalize=True,
    clip_and_rescale=False,
    random_crop=True,
    center_crop=False
)

# Configuration for Patch Embedding
config = {
    'image_height': 64,
    'image_width': 64,
    'im_channels': 12,
    'emb_dim': 768,
    'patch_emb_drop': 0.1,
    'patch_height': 4,
    'patch_width': 4,
}

# Initialize Patch Embedding
patch_embedding = PatchEmbedding(config)

# Get a batch of data
for inputs, labels in train_dataset:
    inputs_emb = patch_embedding(inputs)
    print(f"Input embeddings shape: {inputs_emb.shape}")
    break

# Visualize some input images and their embeddings
def plot_images_and_embeddings(inputs, embeddings, num_images=4):
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 20))
    for i in range(num_images):
        img = inputs[i].permute(1, 2, 0).numpy()
        emb = embeddings[i].detach().numpy()
        axs[i, 0].imshow(img[:, :, 0], cmap='gray')
        axs[i, 0].set_title(f"Input Image {i+1}")
        axs[i, 1].imshow(emb, cmap='viridis')
        axs[i, 1].set_title(f"Embedding {i+1}")
    plt.show()

# Plot some images and embeddings
plot_images_and_embeddings(inputs, inputs_emb)
