import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from attention import MultiHeadSelfAttention
# from patchEmbedding import patch_embedding
# from preprocess_data import train_dataset

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(emb_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        attn_output, attention_weights = self.self_attn(x)
        x = x + attn_output
        x = self.norm1(x)
        
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        
        return x, attention_weights
