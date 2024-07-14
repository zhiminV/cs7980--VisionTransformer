import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocess_data import get_dataset
from patchEmbedding import PatchEmbedding
from attention import MultiHeadSelfAttention
from transformerLayer import TransformerEncoderLayer

class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes=1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(config['emb_dim'], config['num_heads'], config['mlp_dim'], config['dropout_rate'])
            for _ in range(config['num_layers'])
        ])
        self.norm = nn.LayerNorm(config['emb_dim'])
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(config['emb_dim'], 512, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.transformer_layers:
            x, _ = layer(x)
        x = self.norm(x)
        x = x[:, 1:]  # Remove class token
        # Reshape to 2D feature map
        n, l, c = x.shape
        h = w = int(l ** 0.5)
        x = x.permute(0, 2, 1).view(n, c, h, w)
        # Upsample to original size
        x = self.upsample(x)
        return x

