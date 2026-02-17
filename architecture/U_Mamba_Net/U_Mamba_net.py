import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.U_Mamba_Net.U_Mamba_blocks import Full_U_Mamba_Block, Residual_Block, Strided_Conv, Transposed_Conv

class U_Mamba_net(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, ssm_dim=8, num_classes=1, use_checkpointing=False): #num_classes is out_channels
        super(U_Mamba_net, self).__init__()

        mamba_block_args = {"ssm_dim": ssm_dim, "use_checkpointing": use_checkpointing}

        #Encoder path
        self.enc1_down = Strided_Conv(in_channels, base_channels)
        self.enc1_mamba = Full_U_Mamba_Block(base_channels, **mamba_block_args)

        self.enc2_down = Strided_Conv(base_channels, base_channels * 2)
        self.enc2_mamba = Full_U_Mamba_Block(base_channels * 2, **mamba_block_args)

        self.enc3_down = Strided_Conv(base_channels * 2, base_channels * 4)
        self.enc3_mamba = Full_U_Mamba_Block(base_channels * 4, **mamba_block_args)

        self.enc4_down = Strided_Conv(base_channels * 4, base_channels * 8)
        self.enc4_mamba = Full_U_Mamba_Block(base_channels * 8, **mamba_block_args)

        #Bottleneck
        self.bottleneck_mamba = Full_U_Mamba_Block(base_channels * 8, **mamba_block_args)
        self.bottleneck_res = Residual_Block(base_channels * 8)

        #Decoder path
        self.dec4_up = Transposed_Conv(base_channels * 8, base_channels * 4)
        self.dec4_mamba = Full_U_Mamba_Block(base_channels * 4, **mamba_block_args)
        self.dec4_res = Residual_Block(base_channels * 4)

        self.dec3_up = Transposed_Conv(base_channels * 4, base_channels * 2)
        self.dec3_mamba = Full_U_Mamba_Block(base_channels * 2, **mamba_block_args)
        self.dec3_res = Residual_Block(base_channels * 2)
        self.aux3_head = nn.Conv3d(base_channels * 2, num_classes, kernel_size=1, bias=True)

        self.dec2_up = Transposed_Conv(base_channels * 2, base_channels)
        self.dec2_mamba = Full_U_Mamba_Block(base_channels, **mamba_block_args)
        self.dec2_res = Residual_Block(base_channels)
        self.aux2_head = nn.Conv3d(base_channels, num_classes, kernel_size=1, bias=True)

        self.dec1_up = Transposed_Conv(base_channels, base_channels)
        self.dec1_mamba = Full_U_Mamba_Block(base_channels, **mamba_block_args)
        self.dec1_res = Residual_Block(base_channels)

        #final output conv
        self.final = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1_mamba(self.enc1_down(x))         # [B, C, H/2, W/2, D/2]
        #print(x1.shape) #layer 1
        x2 = self.enc2_mamba(self.enc2_down(x1))        # [B, 2C, H/4, W/4, D/4]
        #print(x2.shape) #layer 2
        x3 = self.enc3_mamba(self.enc3_down(x2))        # [B, 4C, H/8, W/8, D/8]
        #print(x3.shape) #layer 3
        x4 = self.enc4_mamba(self.enc4_down(x3))        # [B, 8C, H/16, W/16, D/16]
        #print(x4.shape) #layer 4

        # Bottleneck
        x5 = self.bottleneck_res(self.bottleneck_mamba(x4))
        #print(x5.shape) #bottleneck

        # Decoder
        x = self.dec4_up(x5)
        x = self.dec4_mamba(x + x3)
        x = self.dec4_res(x)
        #print(x.shape) #uplayer 4

        x = self.dec3_up(x)
        x = self.dec3_mamba(x + x2)
        x = self.dec3_res(x)
        aux3 = self.aux3_head(x)  # Deep supervision output 1
        #print(x.shape) #uplayer 3

        x = self.dec2_up(x)
        x = self.dec2_mamba(x + x1)
        x = self.dec2_res(x)
        aux2 = self.aux2_head(x)  # Deep supervision output 2
        #print(x.shape) #uplayer 2

        x = self.dec1_up(x)
        x = self.dec1_mamba(x)
        x = self.dec1_res(x)
        #print(x.shape) #uplayer 1

        final = self.final(x)

        main_size = final.shape[2:]  # (H, W, D)
        aux2 = F.interpolate(aux2, size=main_size, mode='trilinear', align_corners=False)
        aux3 = F.interpolate(aux3, size=main_size, mode='trilinear', align_corners=False)

        return final, aux2, aux3

""" if __name__ == "__main__":
    # Dummy input: batch size 1, 1 input channel, 64x64x64 volume
    x = torch.randn(1, 1, 64, 64, 64)

    # Initialize model
    model = U_Mamba_net(in_channels=1, base_channels=32, ssm_dim=64, num_classes=3)

    # Run forward pass
    with torch.no_grad():
        y = model(x)

    # Print output shape
    print("Input shape :", x.shape)
    print("Output shape:", y.shape) """
