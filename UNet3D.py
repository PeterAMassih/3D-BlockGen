import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention block for 3D
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.shape
        query = self.query(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, D * H * W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(batch_size, -1, D * H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)

        return self.gamma * out + x # TODO check if residual connection needed

# Residual Block for 3D
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # TODO change dim of conv not to have mismatch no need to change dim of shortcut
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# UNet3D with Residual and Self-Attention Blocks
class UNet3DWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3DWithAttention, self).__init__()
        
        features = init_features
        self.encoder1 = ResidualBlock3D(in_channels, features)
        self.encoder2 = ResidualBlock3D(features, features * 2)
        self.encoder3 = ResidualBlock3D(features * 2, features * 4)
        self.encoder4 = ResidualBlock3D(features * 4, features * 8)

        # Self-attention after last encoder block
        self.attention = SelfAttention3D(features * 8)

        self.bottleneck = ResidualBlock3D(features * 8, features * 16)

        # Decoder blocks with upsampling
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock3D(features * 8 + features * 8, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock3D(features * 4 + features * 4, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock3D(features * 2 + features * 2, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock3D(features + features, features)

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        # Apply attention
        enc4 = self.attention(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2, stride=2))

        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
