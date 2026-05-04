import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    # use both average and max pooling merged
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        outputs = []
        for ppm in self:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(ppm_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            outputs.append(ppm_out)
        return outputs


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=13):
        super(PPMHEAD, self).__init__()
        self.psp_modules = PPM(pool_sizes, in_channels, out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=256):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)
        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        x1 = self.PPMHead(input_fpn[-1])

        x = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        # new: residual fusion refinement
        x1 = F.interpolate(x1, x4.shape[-2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, x4.shape[-2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, x4.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x1, x2, x3, x4], 1)
        x = self.fuse_all(fused) + x4  # add residual skip from low-level

        return x


class UPerNet(nn.Module):
    def __init__(self, num_classes=13, image_size=128, debug=False, embed_dim=768):
        super(UPerNet, self).__init__()
        self.num_classes = num_classes
        self.img_size = image_size
        self.decoder = FPNHEAD()

        # In UPerNet.__init__ (restore stride arguments)

        self.conv0 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=8, stride=8),  # -> 128x128
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=4),  # -> 64x64
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),  # -> 32x32
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.Dropout(0.5)  # stays 16x16
        )


        self.cls_seg = nn.Conv2d(256, self.num_classes, 3, padding=1)

    def forward(self, features):
        x = F.interpolate(features, size=(16, 16), mode='bilinear', align_corners=False)

        m = [self.conv0(x), self.conv1(x), self.conv2(x), self.conv3(x)]
        x = self.decoder(m)

        x = self.cls_seg(x)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x
