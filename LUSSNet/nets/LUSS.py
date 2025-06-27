import torch.nn as nn
import math
import torch
import warnings
from einops import rearrange
import torch.nn.functional as F
from torchstat import stat
import numbers
from torch.nn import init

class FeatureAdjuster(nn.Module):
    def __init__(self):
        super(FeatureAdjuster, self).__init__()
        self.relu_activation = nn.ReLU()

    def forward(self, feature_a, feature_b):

        shape_a, shape_b = feature_a.size(), feature_b.size()

        assert shape_a[1] == shape_b[1]

        cosine_similarity = F.cosine_similarity(feature_a, feature_b, dim=1)

        cosine_similarity = cosine_similarity.unsqueeze(1)

        feature_a = feature_a + feature_b * cosine_similarity

        feature_a = self.relu_activation(feature_a)

        return feature_a


class PositionEncodingModule(nn.Module):
    def __init__(self, channels, direction, window_size):
        super().__init__()
        self.direction = direction
        self.channels = channels
        self.window_size = window_size


        if self.direction == 'H':
            self.pos_encoding = nn.Parameter(torch.randn(1, channels, window_size, 1))
        elif self.direction == 'W':
            self.pos_encoding = nn.Parameter(torch.randn(1, channels, 1, window_size))


        init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, feature_map):

        pos_enc_expanded = self.pos_encoding.expand(1, self.channels, self.window_size, self.window_size)

        return feature_map + pos_enc_expanded

def autopad(k, p=None, d=1):

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  #
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv1(nn.Module):

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(Conv1,self).__init__()
        self.c1=c1
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))





class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized
class Channel_aifengheguai(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = self.dconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6
class Spatial_aifengheguai(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class FGMLFA(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()

        self.c_ = int(c2 / 4)  # hidden channels
        self.lraynoram = LayerNorm2D(c1)
        self.c1 = nn.Conv2d(self.c_, self.c_, 3, 1, 1, 1 )
        self.c2 = nn.Conv2d(int(self.c_ * 2), self.c_, 3, 1, 3, 3)
        self.bn0 = nn.BatchNorm2d(self.c_)
        self.c3 = nn.Conv2d(int(self.c_ * 2), self.c_, 3, 1, 5, 5)
        self.c = nn.Conv2d(c1, c2, 3, 1, 1)
        self.bn = nn.BatchNorm2d(c1)

        self.reul = nn.GELU()
        self.add = shortcut and c1 == c2
        self.spatial_aifhg = Spatial_aifengheguai( self.c_)
        self.channel_aifhg = Channel_aifengheguai(self.c_)
        self.c11 = nn.Conv2d(self.c_ * 2, self.c_, 1, 1, )
    def forward(self, x):
        id = x

        x = self.lraynoram(x)

        x1, x2, x3,x4 = torch.chunk(x, 4, dim=1)

        x1 = self.c1(x1)
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.c2(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self.c3(x3)
        x33 = self.spatial_aifhg(x4) * x3
        x44 = self.channel_aifhg(x3) * x4
        x4=torch.cat([x33, x44], dim=1)
        x4=self.c11(x4)
        x_out = torch.cat((x1, x2, x3,x4), dim=1)
        out = self.c(x_out)

        out = self.bn(out)

        out = self.reul(out)


        return id + out if self.add else out

class Partial_conv1(nn.Module):
    def __init__(self, dim, c2,n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.c1 = c2 - self.dim_conv3
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  # 创建3x3卷积层，不带偏置
        self.conv1 = nn.Conv2d((self.dim_untouched), self.c1, 1, 1)

        self.bn = nn.BatchNorm2d(self.c1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):


        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.bn(self.conv1(x2))
        x = torch.cat((x1, x2), 1)


        return x

    def forward_slicing(self, x):

        x = x.clone()
        # self.dim_conv3      16   [0~16]
        x[:, :self.dim_conv4, :, :] = self.partial_conv3(x[:, :self.dim_conv4, :, :])

        return x

class Perception_block(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()

        self.channel_split_ratio = torch.tensor(e)

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv1(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv1((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(FGMLFA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in
                               range(n))

    def forward(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv1(c1, c_, 1, 1)
        self.cv2 = Conv1(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Partial_conv1(c1 * 4, c2)


    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h horizontal
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v vertical
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d diagonal
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class WFD(nn.Module):
    def __init__(self, dim_in, dim, need=False):
        super(WFD, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1, padding=0)
            self.HaarWavelet = HaarWavelet(dim, grad=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False)
            self.dim = dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)

        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        return a + (h + v + d)


class Backbone(nn.Module):
    def __init__(self, ):
        super(Backbone, self).__init__()
        self.pos = PositionEncodingModule(channels=16, direction='W', window_size=320)
        self.Foucus = Focus(3, 16)
        self.PB_0 = Perception_block(16, 16)
        self.conv1 = Conv1(16, 32, 3, 2, 1)
        self.PB_1 = Perception_block(32, 32)
        self.conv2 = Conv1(32, 64, 3, 2, 1)
        self.PB_2_1 = Perception_block(64, 64, n=1, shortcut=True, )
        self.conv3 = Conv1(64, 128, 3, 2, 1)
        self.PB_3_1 = Perception_block(128, 128, n=1, shortcut=True)
        self.conv4 = Conv1(128, 256, 3, 2, 1)
        self.PB_4_1 = Perception_block(256, 256)
        self.sppf = SPPF(256, 256)
        self.d1 = WFD(3, 3, need=True)

    def forward(self, x):
        d = x
        x = self.Foucus(x)
        x = self.pos(x)
        p1 = self.PB_0(x)
        x = self.conv1(p1)
        p2 = self.PB_1(x)
        x = self.conv2(p2)
        p3 = self.PB_2_1(x)
        x = self.conv3(p3)
        p4 = self.PB_3_1(x)
        x = self.conv4(p4)

        x = self.PB_4_1(x)

        p5 = self.sppf(x)

        d1 = self.d1(d)
        d2 = self.d1(d1)
        d3 = self.d1(d2)
        d4 = self.d1(d3)
        d5 = self.d1(d4)

        return p1, p2, p3, p4, p5, d1, d2, d3, d4, d5




class AFFM(nn.Module):
    def __init__(self, in_channel_num, channel_num):
        super().__init__()
        self.in1 = int(in_channel_num)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False),

            nn.ReLU(inplace=True),
        )
        self.f1= nn.Sequential(
            nn.Conv2d(in_channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False),

            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channel_num, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(in_channel_num)
        self.re = nn.ReLU()
        self.w = nn.Parameter(torch.ones(in_channel_num, dtype=torch.float32), requires_grad=True)  # 可学习的权重参数，初始化为1
        self.epsilon = 0.0001  # 防止除零的小值
        self.fcouns=FeatureAdjuster()
    def forward(self, x1, x2, d):
        x2 = self.up(x2)
        N1, C1, H1, W1 = x1.size()  # 获取第一个输入的维度
        N2, C2, H2, W2 = x2.size()  # 获取第二个输入的维度
        N3, C3, H3, W3 = d.size()
        C2_E = C1 + C2
        w = self.w[:(C1 + C2 + C3)]
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x_c1 = torch.cat([x1, x2, d], dim=1)
        x0uy=self.f1(x_c1)
        x1_1 = (weight[:C1] * x1.view(N1, H1, W1, C1)).view(N1, C1, H1, W1)  # 对第一个输入进行加权
        x2_2 = (weight[C1:C2_E] * x2.view(N2, H2, W2, C2)).view(N2, C2, H2, W2)  # 对第二个输入进行加权
        d_1 = (weight[C1 + C2:] * d.view(N3, H3, W3, C3)).view(N3, C3, H3, W3)  # 对第二个输入进行加权
        # print(x1.size(),x2.size(),d.size())

        x_cat = torch.cat([x1_1, x2_2, d_1], dim=1)
        x_fused=self.fcouns(x_c1,x_cat)
        x_fused = self.fusion_conv(x_fused)
        return x_fused+x0uy


class LUSS(nn.Module):
    def __init__(self, num_classes=8):
        super(LUSS, self).__init__()
        self.Backbone = Backbone()

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, num_classes, 3, 1, 1)


        in_filters = [83, 163, 323, 387]
        out_filters = [32, 64, 128, 256]

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up4 = AFFM(in_filters[3], out_filters[3])
        self.up3 = AFFM(in_filters[2], out_filters[2])
        self.up2 = AFFM(in_filters[1], out_filters[1])
        self.up1 = AFFM(in_filters[0], out_filters[0])

    def forward(self, x):
        [feat1, feat2, feat3, feat4, feat5, d1, d2, d3, d4, d5] = self.Backbone.forward(x)

        # 40*40
        up4 = self.up4(feat4, feat5, d4)

        up3 = self.up3(feat3, up4, d3)

        up2 = self.up2(feat2, up3, d2)

        up1 = self.up1(feat1, up2, d1)

        up1 = self.up_conv(up1)

        final = self.final(up1)

        return final


if __name__ == '__main__':
    net = LUSS(num_classes=8)
    # x=net(a)
    stat(net, (3, 640, 640))