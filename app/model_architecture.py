import torch
import torch.nn as nn
from torchvision import models


DEFAULT_IMAGE_SIZE = 64

LWVIT_PATCH_SIZE = 8
LWVIT_EMBED_DIM = 128
LWVIT_DEPTH = 4
LWVIT_NUM_HEADS = 4
LWVIT_MLP_DIM = 256
LWVIT_DROPOUT = 0.1

PFVIT_CNN_STEM_CHANNELS = 64
PFVIT_PATCH_SIZE = 4
PFVIT_EMBED_DIM = 128
PFVIT_DEPTH = 4
PFVIT_NUM_HEADS = 4
PFVIT_MLP_DIM = 256
PFVIT_DROPOUT = 0.1


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.20),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_simple_cnn_model(num_classes):
    return SimpleCNN(num_classes=num_classes)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout_rate=0.05),
            ConvBlock(32, 64, dropout_rate=0.10),
            ConvBlock(64, 128, dropout_rate=0.15),
            ConvBlock(128, 256, dropout_rate=0.20),
            ConvBlock(256, 384, dropout_rate=0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_improved_cnn_model(num_classes):
    return ImprovedCNN(num_classes=num_classes)


def create_resnet18_model(num_classes):
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


class LightweightViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout):
        super(LightweightViT, self).__init__()

        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = self.transformer(x)

        cls_output = self.norm(x[:, 0])
        logits = self.classifier(cls_output)

        return logits


def create_lwvit_model(
    num_classes,
    image_size=DEFAULT_IMAGE_SIZE,
    patch_size=LWVIT_PATCH_SIZE,
    embed_dim=LWVIT_EMBED_DIM,
    depth=LWVIT_DEPTH,
    num_heads=LWVIT_NUM_HEADS,
    mlp_dim=LWVIT_MLP_DIM,
    dropout=LWVIT_DROPOUT
):
    return LightweightViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    )


class CNNStem(nn.Module):
    def __init__(self, out_channels):
        super(CNNStem, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.stem(x)


class PFViT(nn.Module):
    def __init__(
        self,
        image_size,
        num_classes,
        cnn_stem_channels,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_dim,
        dropout
    ):
        super(PFViT, self).__init__()

        stem_output_size = image_size // 2
        assert stem_output_size % patch_size == 0

        self.num_patches = (stem_output_size // patch_size) ** 2

        self.cnn_stem = CNNStem(out_channels=cnn_stem_channels)

        self.patch_embed = nn.Conv2d(
            in_channels=cnn_stem_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.cnn_stem(x)
        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = self.transformer(x)

        cls_output = x[:, 0]
        logits = self.classifier(cls_output)

        return logits


def create_pfvit_model(
    num_classes,
    image_size=DEFAULT_IMAGE_SIZE,
    cnn_stem_channels=PFVIT_CNN_STEM_CHANNELS,
    patch_size=PFVIT_PATCH_SIZE,
    embed_dim=PFVIT_EMBED_DIM,
    depth=PFVIT_DEPTH,
    num_heads=PFVIT_NUM_HEADS,
    mlp_dim=PFVIT_MLP_DIM,
    dropout=PFVIT_DROPOUT
):
    return PFViT(
        image_size=image_size,
        num_classes=num_classes,
        cnn_stem_channels=cnn_stem_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    )
