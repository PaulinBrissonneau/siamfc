#Code forké depuis "huanglianghua" (https://github.com/huanglianghua/siamfc-pytorch)
#Adapté et modifié par Paulin Brissonneau

"""
Différentes architectures utilisables en tant que backbone.
Pour le projet, les architectures "AlexNetImp", "ResnetSEG", "ResnetCLA", sont des architectures qui ont été testées dans le cadre du projet.
L'idée été de tester différents transfert d'apprentissage. Ca n'a pas été concluant, ça n'apparait pas dans le rapport. 
"""

import torchvision
import torchvision.models as models

import torch.nn as nn

class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class _AlexNet(nn.Module):
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

"""AlexNet pré-entrainé en classification"""
class AlexNetImp(nn.Module):
    def __init__ (self, pretrained) :
        super(AlexNetImp, self).__init__()
        imp = models.alexnet(pretrained=pretrained, progress=True)
        self.backbone = nn.Sequential(*list(imp.features.children())[:-2])
        self.backbone[-1] = nn.Conv2d(256, 256, 3, 1)

    def forward (self, x):
        x = self.backbone(x)
        return x

"""AlexNet pré-entrainé en segmentation"""
class ResnetSEG(nn.Module):

    def __init__ (self, pretrained) :
        super(ResnetSEG, self).__init__()
        imp = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
        self.backbone = nn.Sequential(*list(imp.children())[:-2])
        print(len([mod for mod in list(self.backbone.modules()) if hasattr(mod, 'reset_parameters')]))
        self.conv = nn.Conv2d(2048, 256, 3, 1)

    def forward (self, x):
        x = self.backbone(x)['out']
        x = self.conv(x)
        return x
        
"""AlexNet pré-entrainé en classification"""
class ResnetCLA(nn.Module):

    def __init__ (self, pretrained) :
        super(ResnetCLA, self).__init__()
        imp = torchvision.models.resnet101(pretrained=pretrained, progress=True)
        self.backbone = nn.Sequential(*list(imp.children())[:-2])
        self.conv = nn.Conv2d(2048, 256, 3, 1)
        
    def forward (self, x):
        x = self.backbone(x)
        x = self.conv(x)
        return x

class AlexNetV0(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV0, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))