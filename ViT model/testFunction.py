import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocess_data import get_dataset
from patchEmbedding import PatchEmbedding
from attention import MultiHeadSelfAttention
from transformerLayer import TransformerEncoderLayer

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

# Initialize MultiHeadSelfAttention
attention_layer = MultiHeadSelfAttention(emb_dim=768, num_heads=8, dropout_rate=0.1)

# Initialize TransformerEncoderLayer
transformer_layer = TransformerEncoderLayer(emb_dim=768, num_heads=8, mlp_dim=2048, dropout_rate=0.1)

# Constants
DATA_SIZE = 64
PATCH_SIZE = 64  # Can change as needed; in Bronte's paper, she made it 32
BATCH_SIZE = 32

# Load dataset
train_dataset = get_dataset(
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

# Print an example batch to check shapes
for inputs, labels in train_dataset:
    print(f"Example batch input shape: {inputs.shape}, Example batch number of channels: {inputs.shape[-1]}")
    break

# Convert TensorFlow dataset to PyTorch tensors and process them
for tf_inputs, tf_labels in train_dataset:
    # Convert TensorFlow tensors to NumPy arrays, then to PyTorch tensors
    inputs = torch.from_numpy(tf_inputs.numpy()).permute(0, 3, 1, 2)  # Convert (batch_size, height, width, channels) to (batch_size, channels, height, width)
    labels = torch.from_numpy(tf_labels.numpy()).permute(0, 3, 1, 2)

    # Pass the inputs through the patch embedding layer
    inputs_emb = patch_embedding(inputs)
    print(f"Input embeddings shape: {inputs_emb.shape}")
    
    # Pass the embeddings through the attention layer
    attention_output, attention_weights = attention_layer(inputs_emb)
    print(f"Attention output shape: {attention_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Pass the embeddings through the transformer encoder layer
    transformer_output, transformer_attention_weights = transformer_layer(inputs_emb)
    print(f"Transformer output shape: {transformer_output.shape}")
    print(f"Transformer attention weights shape: {transformer_attention_weights.shape}")

    # Visualize some input images and their embeddings
    def plot_images_and_embeddings(inputs, embeddings, num_images=2):
        fig, axs = plt.subplots(num_images, 2, figsize=(10, 20))
        for i in range(num_images):
            img = inputs[i].permute(1, 2, 0).numpy()
            emb = embeddings[i].detach().numpy()
            axs[i, 0].imshow(img[:, :, 0], cmap='viridis')
            axs[i, 0].set_title(f"Input Image {i+1}")
            axs[i, 1].imshow(emb, cmap='viridis')
            axs[i, 1].set_title(f"Embedding {i+1}")
        plt.show()

    # Visualize some attention weights
    def plot_attention_weights(attention_weights, num_images=2):
        fig, axs = plt.subplots(num_images, 1, figsize=(10, 10))
        for i in range(num_images):
            att = attention_weights[i, 0].detach().numpy()  # Visualizing the attention weights for the first head

            # Plot attention weights as heatmap
            axs[i].imshow(att, aspect='auto', cmap='viridis')
            axs[i].set_title(f"Attention Weights {i+1}")
            axs[i].axis('off')
        plt.show()

    # Plot some images and embeddings
    plot_images_and_embeddings(inputs, inputs_emb)
    # Plot some attention weights
    plot_attention_weights(attention_weights)
    # Plot some attention weights from the transformer encoder layer
    plot_attention_weights(transformer_attention_weights)
    break
