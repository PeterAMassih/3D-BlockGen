import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    """
    Timestep embedding module for conditioning the model on diffusion timesteps.
    """
    def __init__(self, time_embed_dim):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(1, time_embed_dim)  # 1 input feature since timesteps are scalars
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, timesteps):
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)  # Convert shape from (batch_size,) -> (batch_size, 1)

        emb = self.linear_1(timesteps.float())  # Convert timesteps to float for embedding
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class AttentionBlock3D(nn.Module):
    """
    Self-attention mechanism for 3D data using multi-head attention.
    """
    def __init__(self, channels, num_heads):
        super(AttentionBlock3D, self).__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, D * H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, C, D, H, W)
        out = self.proj_out(out)
        return x + out


class ResnetBlock3D(nn.Module):
    """
    3D Residual block with optional time embedding integration.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim=None):
        super(ResnetBlock3D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        if time_embed_dim is not None:
            self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        else:
            self.time_emb_proj = None

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = F.silu(h)  
        h = self.conv1(h)

        if temb is not None:
            temb = self.time_emb_proj(temb)
            h = h + temb[:, :, None, None, None]  # Add the time embedding projection

        h = self.norm2(h)
        h = F.silu(h)  
        h = self.conv2(h)

        return self.skip(x) + h


class DownBlock3D(nn.Module):
    """
    Downsampling block in UNet3D with residual connections and optional attention.
    """
    def __init__(self, in_channels, out_channels, num_layers, time_embed_dim=None, attention_head_dim=None):
        super(DownBlock3D, self).__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels, time_embed_dim) for i in range(num_layers)])
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.attention = AttentionBlock3D(out_channels, attention_head_dim) if attention_head_dim is not None else None

    def forward(self, x, temb=None):
        for resnet in self.resnets:
            x = resnet(x, temb)
        if self.attention:
            x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock3D(nn.Module):
    """
    Upsampling block in UNet3D with residual connections and optional attention.
    """
    def __init__(self, in_channels, out_channels, num_layers, time_embed_dim=None, attention_head_dim=None):
        super(UpBlock3D, self).__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels, time_embed_dim) for i in range(num_layers)])
        self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention = AttentionBlock3D(out_channels, attention_head_dim) if attention_head_dim is not None else None

    def forward(self, x, skip, temb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        for resnet in self.resnets:
            x = resnet(x, temb)
        if self.attention:
            x = self.attention(x)
        return x


class MidBlock3D(nn.Module):
    """
    MidBlock3D with residual connections and attention at the bottleneck.
    """
    def __init__(self, in_channels, time_embed_dim=None, attention_head_dim=None):
        super(MidBlock3D, self).__init__()
        self.resnet1 = ResnetBlock3D(in_channels, in_channels, time_embed_dim)
        self.attention = AttentionBlock3D(in_channels, attention_head_dim) if attention_head_dim is not None else None
        self.resnet2 = ResnetBlock3D(in_channels, in_channels, time_embed_dim)

    def forward(self, x, temb=None):
        x = self.resnet1(x, temb)
        if self.attention:
            x = self.attention(x)
        x = self.resnet2(x, temb)
        return x


class UNet3D(nn.Module):
    """
    3D UNet architecture for diffusion models with time embedding and attention.
    """
    def __init__(
        self,
        in_channels=1,  # Input channels (e.g., 1 for grayscale/voxel data)
        out_channels=1,  # Output channels
        base_channels=64,  # Number of filters in the base layer
        layers_per_block=2,  # Number of ResNet layers per block
        block_out_channels=(64, 128, 256, 512),  # Number of output channels per block
        time_embed_dim=256,  # Dimension of time embedding
        attention_head_dim=8,  # Attention head dimension for multi-head attention
    ):
        super(UNet3D, self).__init__()

        # Initial input convolution
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        # Time embedding
        self.time_embedding = TimestepEmbedding(time_embed_dim)

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        prev_channel = base_channels
        for out_channel in block_out_channels:
            self.down_blocks.append(
                DownBlock3D(prev_channel, out_channel, layers_per_block, time_embed_dim, attention_head_dim)
            )
            prev_channel = out_channel

        # Mid block
        self.mid_block = MidBlock3D(block_out_channels[-1], time_embed_dim, attention_head_dim)

        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_out_channels))
        for i, out_channel in enumerate(reversed_channels):
            prev_channel = reversed_channels[i-1] if i > 0 else block_out_channels[-1]
            self.up_blocks.append(
                UpBlock3D(prev_channel + out_channel, out_channel, layers_per_block, time_embed_dim, attention_head_dim)
            )

        # Final output convolution
        self.conv_out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps):
        """
        Forward pass through the 3D UNet.

        Args:
            x: Input tensor of shape (batch_size, channels, D, H, W)
            timesteps: Timestep tensor for diffusion conditioning
        """
        # Ensure timesteps are correctly shaped
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)  # Now timesteps is (batch_size, 1)

        # Initial convolution
        x = self.conv_in(x)

        # Time embedding
        temb = self.time_embedding(timesteps)  # Properly expand the timesteps

        # Downsample with skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, temb)
            skips.append(skip)

        # Mid block
        x = self.mid_block(x, temb)

        # Upsample with skip connections
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[-(i+1)]
            x = up_block(x, skip, temb)

        # Final output convolution
        x = self.conv_out(x)
        return x
