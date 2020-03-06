import torch
from torch import nn
import torch.nn.functional as F

import models.base_resnet101 as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class AFENet(nn.Module):
    def __init__(self, classes=19, pretrained_model_path=None):
        super(AFENet, self).__init__()
        bins = (1, 2, 3, 6)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        resnet = models.resnet101(pretrained_model_path=pretrained_model_path)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        fea_dim *= 2

        self.ppm_reduce = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aff1 = AFEM(256, 1024, 256)
        self.aff2 = AFEM(1024, 512, 256)
        self.aff3 = AFEM(512, 256, 256, l_stride=2)

        self.cls = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        imsize = x.size()[2:]

        x = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        x_temp = self.layer3(c2)
        c4 = self.layer4(x_temp)

        c4 = self.ppm(c4)
        c4 = self.ppm_reduce(c4)
        c1 = self.aff3(c2, c1)
        c2 = self.aff2(x_temp, c2)
        c3 = self.aff1(c4, x_temp)

        c3 = c4 + c3
        c2 = c3 + c2
        c1 = c2 + c1

        x = torch.cat([c1, c2, c3, c4], dim=1)

        x = self.cls(x)
        x = F.interpolate(x, size=imsize, mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_temp)
            aux = F.interpolate(aux, size=imsize, mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


class AFEM(nn.Module):
    def __init__(self, h_in_channel, l_in_channel, out_channel=256, l_stride=1):
        super(AFEM, self).__init__()
        self.conv_H = nn.Sequential(nn.Conv2d(h_in_channel, out_channel, 1, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.conv_L = nn.Sequential(nn.Conv2d(l_in_channel, out_channel, 1, stride=l_stride, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.conv_A = nn.Sequential(nn.Conv2d(out_channel, out_channel, 1, bias=False),
                                    nn.Softmax(dim=-1))
        self.conv = nn.Sequential(nn.Conv2d(3*out_channel, out_channel, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, h_x, l_x):
        h_x = self.conv_H(h_x)
        l_x = self.conv_L(l_x)

        attention = self.conv_A(l_x)
        attention = attention.mean(1, keepdim=True)
        attention = h_x * attention

        h_x = h_x + attention
        f_la = l_x + attention

        output = torch.cat([h_x, f_la, l_x], dim=1)
        output = self.conv(output)

        return output


if __name__ == '__main__':
    inputs = torch.rand(4, 3, 768, 768).cuda()
    model = AFENet(classes=19, pretrained_model_path=None).cuda()
    model.eval()
    print(model)
    output = model(inputs)
    print('AFENet', output.size())
