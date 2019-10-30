import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)


class HPM(nn.Module):
    def __init__(self, in_dim, out_dim, bin_level_num=5):
        super(HPM, self).__init__()
        self.bin_num = [2**i for i in range(bin_level_num)]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform(
                    torch.zeros(sum(self.bin_num), in_dim, out_dim)))])
    def forward(self, x):
        feature = list()
        n, c, h, w = x.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3)+z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        
        feature = feature.matmul(self.fc_bin[0])
        return feature.permute(1, 0, 2).contiguous()