from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision.models as models



transform = transforms.Compose([
    transforms.ToTensor()
])


#model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, num_classes=21, aux_loss=None)


#model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True)
"""https://neurohive.io/en/popular-networks/resnet/"""



model = models.alexnet(pretrained=True)
#print(model)

#for module in list(model.features.modules()) [:-1] :
#    print(module)

#print(model)

#model = torchvision.models.resnet101(pretrained=True, progress=True)
print(model)
#m2 = nn.Sequential(*list(model.children())[:-2])
#print(m2)
#print(len(list(m2.modules())))
#print(len([mod for mod in list(m2.modules()) if isinstance(mod, nn.Conv2d)]))
#print(len([mod for mod in list(m2.modules()) if hasattr(mod, 'reset_parameters')]))

#model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True)
#m2 = nn.Sequential(*list(model.children())[:-2])
#print(m2)
#print(len(list(m2.modules())))
#print(len([mod for mod in list(m2.modules()) if isinstance(mod, nn.Conv2d)]))
#print(len([mod for mod in list(m2.modules()) if hasattr(mod, 'reset_parameters')]))

#print(len(list(model.backbone.parameters())))
#print(len(list(model.parameters())))
#print(len(list(model.classifier.parameters())))
#print(len(list(model.aux_classifier.parameters())))
#print(model.classifier.modules())
#print(len([mod for mod in list(model.backbone.layer3.modules()) if isinstance(mod, nn.Conv2d)]))
#for module in model.backbone.layer3 :
#    print(module)
#print(model.classifier)
#print("--------------------------   model.backbone :")
#print(model.features)
#print("--------------------------   model.classifier :")
#print(model.classifier)
#print(model.layer1.in_features)

#for truc in model.backbone :
#    print(truc)


#cifar = torchvision.datasets.CIFAR10('/home/paulin/Documents/data/', download=True, transform=transform)
#dataloader = torch.utils.data.DataLoader(cifar, batch_size=4,shuffle=True)
"""
for it, batch in enumerate(dataloader) :
    x, y = batch
    print("batch") 
    print(x.shape)
    print(y.shape)
    model.features.eval()
    z = model.features(x)
    print(z.keys())
    a = z['out'][0]
    print(z['out'][0].shape)
    for k in range(15) :
        z = a[k, :, :].detach().numpy()
        print(z.shape)
        plt.imshow(z)
        plt.savefig(f"z{k}.png")
    plt.imshow(x[0][0, :, :])
    print(x[0][0, :, :].shape)
    plt.savefig("x.png")
"""
