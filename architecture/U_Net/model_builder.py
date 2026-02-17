import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
        
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        #encoder
        self.conv1 = DoubleConv3D(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv3D(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv3D(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv3D(256, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5 = DoubleConv3D(512, 1024)

        #decoder
        self.up6 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv3D(1024, 512)
        self.up7 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv3D(512, 256)
        self.up8 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv3D(256, 128)
        self.up9 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv3D(128, 64)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)

    def crop_to_match(self, source, target):
        """Crop source tensor spatially to match target size."""
        diffZ = source.shape[2] - target.shape[2]  # Depth
        diffY = source.shape[3] - target.shape[3]  # Height
        diffX = source.shape[4] - target.shape[4]  # Width

        if diffZ > 0:
            source = source[:, :, :source.shape[2] - diffZ, :, :]
        if diffY > 0:
            source = source[:, :, :, :source.shape[3] - diffY, :]
        if diffX > 0:
            source = source[:, :, :, :, :source.shape[4] - diffX]
        
        return source

    def forward(self,x):
        #encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        #decoder
        u6 = self.up6(c5)
        c4 = self.crop_to_match(c4, u6)  # Crop c4 if needed
        merge6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(merge6)

        u7 = self.up7(c6)
        c3 = self.crop_to_match(c3, u7)  # Crop c3 if needed
        merge7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(merge7)

        u8 = self.up8(c7)
        c2 = self.crop_to_match(c2, u8)  # Crop c2 if needed
        merge8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(merge8)

        u9 = self.up9(c8)
        c1 = self.crop_to_match(c1, u9)  # Crop c1 if needed
        merge9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(merge9)

        output = self.final_conv(c9) 
        return output
    
""" if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 32, 128, 128)  # Input: (Batch, Channels, Depth, Height, Width)
    y = model(x)
    print(y.shape) """
