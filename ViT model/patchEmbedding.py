import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
# from preprocess_data import get_dataset

# Define PatchEmbedding class
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super(PatchEmbedding, self).__init__()
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.im_channels = config['im_channels']
        self.emb_dim = config['emb_dim']
        self.patch_emb_drop = config['patch_emb_drop']
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']
        
        # Calculate the number of patches
        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        
        # Linear layer to project patches to embedding dimension
        self.projection = nn.Linear(self.im_channels * self.patch_height * self.patch_width, self.emb_dim)
        
        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.emb_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(self.patch_emb_drop)
        
    def forward(self, x):
        # Extract patches
        batch_size, channels, height, width = x.shape
        patches = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_height * self.patch_width)
        patches = patches.permute(0, 2, 1, 3)  # Rearrange for linear projection
        
        # Visualization: Reshape patches back to image format for visualization
        patches_image = patches.contiguous().view(batch_size, -1, channels, self.patch_height, self.patch_width)
        self.plot_patches(patches_image)
        
        patches = patches.contiguous().view(batch_size, -1, channels * self.patch_height * self.patch_width)
        
        # Apply linear projection and dropout
        embeddings = self.projection(patches)
        embeddings = self.dropout(embeddings)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_tokens, embeddings), dim=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def plot_patches(self, patches_image):
        num_patches = patches_image.shape[1]
        num_images = min(2, patches_image.shape[0])
        fig, axs = plt.subplots(num_images, num_patches, figsize=(num_patches * 2, num_images * 2))
        for i in range(num_images):
            for j in range(num_patches):
                img = patches_image[i, j].permute(1, 2, 0).detach().numpy()
                axs[i, j].imshow(img[:, :, :3])
                axs[i, j].axis('off')
        plt.show()

# # Configuration for Patch Embedding
# config = {
#     'image_height': 64,
#     'image_width': 64,
#     'im_channels': 12,
#     'emb_dim': 768,
#     'patch_emb_drop': 0.1,
#     'patch_height': 4,
#     'patch_width': 4,
# }

# # Initialize Patch Embedding
# patch_embedding = PatchEmbedding(config)

# # Constants
# DATA_SIZE = 64
# PATCH_SIZE = 64  # Can change as needed; in Bronte's paper, she made it 32
# BATCH_SIZE = 32

# # Load dataset
# train_dataset = get_dataset(
#     file_pattern='/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord',
#     data_size=DATA_SIZE,
#     sample_size=PATCH_SIZE,
#     batch_size=BATCH_SIZE,
#     num_in_channels=12,
#     compression_type=None,
#     clip_and_normalize=True,
#     clip_and_rescale=False,
#     random_crop=True,
#     center_crop=False
# )

# # Convert TensorFlow dataset to PyTorch tensors and process them
# for tf_inputs, tf_labels in train_dataset:
#     # Convert TensorFlow tensors to NumPy arrays, then to PyTorch tensors
#     inputs = torch.from_numpy(tf_inputs.numpy()).permute(0, 3, 1, 2)  # Convert (batch_size, height, width, channels) to (batch_size, channels, height, width)
#     labels = torch.from_numpy(tf_labels.numpy()).permute(0, 3, 1, 2)

#     # Pass the inputs through the patch embedding layer
#     inputs_emb = patch_embedding(inputs)
#     print(f"Input embeddings shape: {inputs_emb.shape}")
    
#     # Visualize some input images and their embeddings
#     def plot_images_and_embeddings(inputs, embeddings, num_images=2):
#         fig, axs = plt.subplots(num_images, 2, figsize=(10, 20))
#         for i in range(num_images):
#             img = inputs[i].permute(1, 2, 0).numpy()
#             emb = embeddings[i].detach().numpy()
#             axs[i, 0].imshow(img[:, :, 0], cmap='viridis')
#             axs[i, 0].set_title(f"Input Image {i+1}")
#             axs[i, 1].imshow(emb, cmap='viridis')
#             axs[i, 1].set_title(f"Embedding {i+1}")
#         plt.show()

#     # Plot some images and embeddings
#     plot_images_and_embeddings(inputs, inputs_emb)
#     break