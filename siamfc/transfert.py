#Ecrit par Paulin Brissonneau

"""
Etape de construction des réseaux extracteurs qui appellent ensuite backbones.py et backbone_exp_layer.py.
Ce script sert de laboratoire pour tester différentes architectures et différents pré-entrainements des backbones.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision

from . import ops
from .backbones import AlexNetV1, AlexNetV0, ResnetSEG, ResnetCLA, AlexNetImp
from .backbones_exp_layer import CNNL1, CNNL2, CNNL3, CNNL4, CNNL5, CNNL6, CNNL7, CNNL8
from .heads import SiamFC

"""
classe générique du système siamois : 2 extracteurs et une tête
"""
class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

#initialisation du réseau de référence
def init_vanilla (cfg, params_summary) :
    net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(cfg.out_scale))
    ops.init_weights(net)
    params_summary["type"] = "vanillaV1"
    return net, net.parameters(), params_summary

#intialisation d'un AlexNet pré-entrainé
#cette fonction permet de tester différents type de transfert learning (elle n'est plus utilisée)
def init_trained_alexnet (cfg, params_summary) :
    pretrained = False
    net = Net(
            backbone=AlexNetImp(pretrained=pretrained),
            head=SiamFC(cfg.out_scale))
    params_summary["type"] = "trained_alexnet_"+str(pretrained)
    freeze_params = []
    training_params = []
    mods = list(net.backbone.backbone.modules())

    """
    paragraphe pour choisir les paramètres entrainables
    """

    #training_params = mods[0][-1].parameters() #descend peu ~40
    training_params = mods[0][-4:].parameters() #bloqué à ~0.69
    #training_params = mods[0][-6:].parameters() #bloqué à ~0.69
    #training_params = mods[0][-9:].parameters() #bloqué à ~0.69

    params_summary["training_params"] = str(mods[0][-4:])

    #for mod in mods[0][:9] :
    #    mod.requires_grad = False
    #    freeze_params.append(str(mod))

    params_summary["freeze_params"] = freeze_params

    return net, training_params, params_summary

#fonction pour reset les poids des modules
def reset(to_reset) :
    for module in to_reset :
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            print(f"Reset layer {module}")

#intialisation d'un ResNet pré-entrainé pour étudier d'influence du type de pré-entraienment
def init_model_task (cfg, params_summary) :

    reset_params = []
    freeze_params=[]

    task = "CLA" #variable pour choisir entre classification et segmentation pour le pré-entrainement

    if task == "CLA" :
        net = Net(
            backbone=ResnetCLA(pretrained=True),
            head=SiamFC(cfg.out_scale))

    elif task =="SEG" :
        net = Net(
            backbone=ResnetSEG(pretrained=True),
            head=SiamFC(cfg.out_scale))

    print(len([mod for mod in list(net.modules()) if hasattr(mod, 'reset_parameters')]))

    to_reset = []
    reset_layer4 = False
    if reset_layer4 :
        if task == "CLA":
            #print(list(list(net.backbone.children())[0])[-1])
            to_reset += list(list(net.backbone.children())[0])[-1].modules()
        if task == "SEG" :
            #print(list(list(net.backbone.backbone.children())[0])[-1])
            to_reset += list(net.backbone.backbone.children())[0].layer4.modules()
        reset_params.append("layer4")
        reset(to_reset)

    to_freeze = []
    freeze_params = []

    print("len(to_freeze) : ", len(to_freeze))

    for param in to_freeze :
        param.requires_grad = False

    training_params = []
    for param in net.parameters() :
        if param not in to_freeze :
            training_params.append(param)
    
    params_summary["task"] = task
    params_summary["reset"] = reset_params
    params_summary["freeze_params"] = freeze_params
     
    return net, training_params, params_summary

#Permet de tester différentes configuration de pré-entrainements de l'extracteur
#On peut freeze et reset certaines couches particulières
def init_model_pretrainedstudy (cfg, params_summary) :

    reset_params = []
    freeze_params=[]

    model = "resnet" #ou alexnet

    if model == "resnet" :
        net = Net(
            backbone=Resnet(pretrained=True),
            head=SiamFC(cfg.out_scale))

    elif model =="alexnet" :
        net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(cfg.out_scale))
        ops.init_weights(net)

    """
    paragraphe de reset
    """

    to_reset = []

        #reset resnet backbone weights
    #to_reset += net.backbone.model.backbone.layer4.modules()
    #reset_params.append("layer4")
    #to_reset += net.backbone.model.backbone.layer3.modules()
    #reset_params.append("layer3")
    #to_reset += net.backbone.model.backbone.layer2.modules()
    #reset_params.append("layer2")
    #to_reset += net.backbone.model.backbone.layer1.modules()
    #reset_params.append("layer1")

    print("len(to_reset) : ", len(to_reset))
    reset(to_reset)

    """
    paragraphe de freeze
    """

    to_freeze = []

    #to_freeze += net.parameters()
    #freeze_params.append("conv")

    #to_freeze += net.backbone.conv.parameters()
    #freeze_params.append("conv")

        #reset resnet head weights
    #to_freeze += net.backbone.model.classifier.parameters()
    #freeze_params.append("classifier")
    
        #reset resnet backbone weights
    #to_freeze += net.backbone.model.backbone.layer4.parameters()
    #freeze_params.append("layer4")
    #to_freeze += net.backbone.model.backbone.layer3.parameters()
    #freeze_params.append("layer3")
    #to_freeze += net.backbone.model.backbone.layer2.parameters()
    #reset_params.append("layer2")
    #to_freeze += net.backbone.model.backbone.layer1.parameters()
    #reset_params.append("layer1")

    print("len(to_freeze) : ", len(to_freeze))

    for param in to_freeze :
        param.requires_grad = False

    training_params = []
    for param in net.parameters() :
        if param not in to_freeze :
            training_params.append(param)

    params_summary["reset"] = reset_params
    params_summary["freeze_params"] = freeze_params

    training_params = net.parameters()
    return net, training_params, params_summary



def init_layer_exp(cfg, params_summary):

    if cfg.n_layer == 1 :
        net = Net(
            backbone=CNNL1(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 2 :
        net = Net(
            backbone=CNNL2(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 3 :
        net = Net(
            backbone=CNNL3(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 4 :
        net = Net(
            backbone=CNNL4(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 5 :
        net = Net(
            backbone=CNNL5(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 6 :
        net = Net(
            backbone=CNNL6(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 7 :
        net = Net(
            backbone=CNNL7(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3

    if cfg.n_layer == 8 :
        net = Net(
            backbone=CNNL8(),
            head=SiamFC(cfg.out_scale))
        lr = 1e-3
        
    ops.init_weights(net)
    params_summary["type"] = "CNNL"+str(cfg.n_layer)
    return net, net.parameters(), params_summary, lr