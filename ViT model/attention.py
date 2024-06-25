import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from patchEmbedding import PatchEmbedding
from preprocess_data import train_dataset

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout_rate):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        assert self.head_dim * num_heads == emb_dim, "Embedding dimension must be divisible by the number of heads"
        
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        
        self.fc_out = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        batch_size, seq_length, emb_dim = x.shape
        
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split the embeddings into self.num_heads different pieces
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.nn.functional.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_dim)
        
        # Apply the final linear layer
        out = self.fc_out(out)
        
        return out, attention

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

# Constants
DATA_SIZE = 64
PATCH_SIZE = 64
BATCH_SIZE = 32

# Example data
# Assuming the inputs are prepared, similar to how it was done in the previous example

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

    # Plot some attention weights
    plot_attention_weights(attention_weights)
    break
