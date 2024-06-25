import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocess_data import get_dataset
from patchEmbedding import PatchEmbedding
from attention import MultiHeadSelfAttention
from transformerLayer import TransformerEncoderLayer

class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(config['emb_dim'], config['num_heads'], config['mlp_dim'], config['dropout_rate'])
            for _ in range(config['num_layers'])
        ])
        self.norm = nn.LayerNorm(config['emb_dim'])
        self.classifier = nn.Linear(config['emb_dim'], num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        attentions = []
        for layer in self.transformer_layers:
            x, att = layer(x)
            attentions.append(att)
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.classifier(cls_token)
        return logits, attentions