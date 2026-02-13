import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer

# 定义 ViT-B_4 模型
class ViT_B_4(nn.Module):
    def __init__(self, num_classes=1000):
        super(ViT_B_4, self).__init__()
        self.embed_dim = 384

        self.model = VisionTransformer(img_size=32, patch_size=4, embed_dim=self.embed_dim, depth=12, num_heads=6, mlp_ratio=4)
        self.model.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x