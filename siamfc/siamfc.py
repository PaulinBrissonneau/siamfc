#Code forké depuis "huanglianghua" (https://github.com/huanglianghua/siamfc-pytorch)
#Adapté et modifié par Paulin Brissonneau

"""
Définition du suiveur (script principal)
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import datetime
import cv2
import sys
import os
import time
import sys
import json
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from . import ops
from .transfert import init_model_pretrainedstudy, init_model_task, init_vanilla, init_trained_alexnet, init_layer_exp
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
import torchvision
import torchvision.transforms as trans
from .ops import show_array


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, output_name=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)

        self.cfg = self.parse_args(**kwargs)
        self.output_name = output_name

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.params_summary = {}

        self.running_mean_delta = 1

        """Choix du type d'expérimentation (et donc du type d'extrcateur)"""
        #self.net, training_params, self.params_summary = init_model_pretrainedstudy(self.cfg, self.params_summary)
        #self.net, training_params, self.params_summary = init_model_task(self.cfg, self.params_summary)
        self.net, training_params, self.params_summary = init_vanilla(self.cfg, self.params_summary)
        #self.net, training_params, self.params_summary = init_trained_alexnet(self.cfg, self.params_summary)
        #self.net, training_params, self.params_summary, lr = init_layer_exp(self.cfg, self.params_summary)

        ops.init_weights(self.net)

        #visualisation du réseau
        print(self.net)

        #ligne à décommenter en cas de modification de la valeur de lr selon l'expérience
        #self.cfg = self.parse_args(**kwargs, initial_lr = lr)
        print("init lr : ", self.cfg.initial_lr)
        
        # load checkpoint if provided
        if net_path is not None:
            print('load')
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(training_params,
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
        

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            #expérimentations
            'output_dir' : 'outputs/siamfc/',
            'n_layer': 5,
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 100,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-3, #1e-3
            'ultimate_lr': 1e-5, #1e-5
            'weight_decay': 5e-4, #5e-4
            'momentum': 0.9, #0.9
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box, test_dir_name, expstep, visualize):
        # set to evaluation mode
        self.net.eval()

        #visualisation de l'image de la cible
        if visualize : show_array(img, "kernel img 1", from_np=True, norm=False, expstep=expstep, dir=test_dir_name)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        #visualisation du noyau
        if visualize : show_array(z, "kernel img 2", from_np=True, norm=False, expstep=expstep, dir=test_dir_name)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        #création du noyau
        self.kernel = self.net.backbone(z)
        #création du noyau glissant, au début égal au noyau
        self.track_kernel = self.kernel

        if visualize : show_array(self.kernel[0][0], "kernel", from_np=False, expstep=expstep, dir=test_dir_name)
    
    @torch.no_grad()
    def update(self, img, test_dir_name, expstep, visualize):

        # set to evaluation mode
        self.net.eval()

        #visualisation de l'image de recherche
        if visualize : show_array(img, "search img", from_np=True, norm=False, expstep=expstep, dir=test_dir_name)

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]

        kernel_x = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        #visualisation de l'image de recherche après avoir crop 
        if visualize :
            i = 0
            for f in self.scale_factors :
                arr = x[i]
                i += 1
                show_array(arr, "search img crop"+str(f), from_np=True, norm=False, expstep=expstep, dir=test_dir_name)

        x = np.stack(x, axis=0)
   
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        kernel_x = torch.from_numpy(kernel_x).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        # extractions des noyaux
        x = self.net.backbone(x)
        kernel_x = self.net.backbone(kernel_x)

        #visualisations des cartes de caractéristiques avant la corrélation
        if visualize : show_array(x[0][0], "search pre-corr", from_np=False, expstep=expstep, dir=test_dir_name)
        if visualize : show_array(self.track_kernel[0][0], "track kernel", from_np=False, expstep=expstep, dir=test_dir_name)
    
        #corrélation pour le noyau
        responses = self.net.head(self.kernel, x)
        #corrélation pour le noyau glissant
        track_responses = self.net.head(self.track_kernel, x)

        #mise à jour du noyau glissant
        gamma = 0.2
        self.track_kernel = self.track_kernel+gamma*(kernel_x-self.track_kernel)
    
        responses = responses.squeeze(1).cpu().numpy()
        track_responses = track_responses.squeeze(1).cpu().numpy()

        #visualisations de la corrélation pour le noyau
        if visualize : show_array(responses[0], "post-corr", from_np=True, expstep=expstep, dir=test_dir_name)

        # upsample des corrélations
        def upsample (responses):
            responses = np.stack([cv2.resize(
                u, (self.upscale_sz, self.upscale_sz),
                interpolation=cv2.INTER_CUBIC)
                for u in responses])
            responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
            responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
            return responses

        #upsample pour corrélation puis corrélation glissante
        responses = upsample (responses)
        track_responses = upsample (track_responses)

        """ Autre expérience sur la mise à jour des corrélations glissante, n'apparait pas dans le rapport :
        beta = 0.7
        track_responses = beta*responses+(1-beta)*track_responses
        """

        #visualisation des différents canaux des corrélations
        if visualize :
            chan1 = responses[0]
            chan2 = responses[1]
            chan3 = responses[2]

            chan1 = chan1-np.min(chan1)
            chan1 = chan1/np.max(chan1)

            chan2 = chan2-np.min(chan2)
            chan2 = chan2/np.max(chan2)

            chan3 = chan3-np.min(chan3)
            chan3 = chan3/np.max(chan3)

            show_array(chan1, "upscaled norm response chanel 1", from_np=True, expstep=expstep, dir=test_dir_name, norm=True)
            show_array(chan2, "upscaled norm response chanel 2", from_np=True, expstep=expstep, dir=test_dir_name, norm=True)
            show_array(chan3, "upscaled norm response chanel 3", from_np=True, expstep=expstep, dir=test_dir_name, norm=True)

        #selection de l'argmax des cartes de corrélations et calcul du décalage de la cible
        def peak (responses):
            # peak scale
            scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
            # peak location
            response = responses[scale_id]
            response -= response.min()
            response /= response.sum() + 1e-16
            response = (1 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hann_window
            loc = np.unravel_index(response.argmax(), response.shape)
            max_val = response.max()
            # locate target center
            disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
            disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up
            disp_in_image = disp_in_instance * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
            return response, loc, disp_in_response, disp_in_instance, disp_in_image, scale_id, max_val


        #argmax pour le noyau
        response, loc, disp_in_response, disp_in_instance, disp_in_image, scale_id, max_val = peak(responses)
        #argmax pour le noyau glissant
        track_response, track_loc, track_disp_in_response, track_disp_in_instance, track_disp_in_image, track_scale_id, track_max = peak(track_responses)

        #position de l'objet d'après le noyau glissant
        self.track_center = self.center + track_disp_in_image

        #états internes du suiveur (cf. partie "travail futur" du rapport)
        norm = np.linalg.norm(disp_in_image)
        track_norm = np.linalg.norm(track_disp_in_image)

        #prise en compte pondérée du noyau glissant
        beta = 0.8
        self.mean_center = self.center + beta*disp_in_image + (1-beta)*track_disp_in_image

        raw_center = self.center + disp_in_image

        """Choix des trois méthodes : center (ref), track_center (noyau glissant), et mean (melange des deux selon beta)"""
        self.center = raw_center
        #self.center = self.mean_center
        #self.center = self.track_center

        #états internes du suiveur (cf. partie "travail futur" du rapport)
        epsi = 0.4
        self.running_mean_delta += epsi*(norm - self.running_mean_delta)

        #visualisation des réponses du suiveur avec noyau de référence, puis noyau glissant
        if visualize :
            response_viz = response-np.min(response)
            response_viz = response_viz/np.max(response_viz)
            show_array(response_viz, "peak response", from_np=True, expstep=expstep, dir=test_dir_name, marker=loc)

            track_response_viz = track_response-np.min(track_response)
            track_response_viz = track_response_viz/np.max(track_response_viz)
            show_array(track_response_viz, "track peak response", from_np=True, expstep=expstep, dir=test_dir_name, marker=loc)

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]

        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box, raw_center, self.track_center, self.mean_center, max_val, track_max, norm, beta, self.running_mean_delta
    

    #fonction de suivi, nécessaire pour utiliser "got10k.experiments"
    #son rôle principal est d'itérer dans "img_files" et appeller "self.update" à chaque étape
    def track(self, img_files, box, visualize=True):

        center=None
        track_center=None
        mean_center=None
        max_val = None
        track_max = None
        norm = None
        beta = None
        running_mean_delta = None

        seq_name = img_files[0].split('/')[-3]

        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        now = datetime.datetime.now()
        
        if self.output_name is None :
            test_dir_name = self.cfg.output_dir+"[TEST]siamfc-"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"-"+str(now.hour)+"_"+str(now.minute)
        else :
            test_dir_name = self.cfg.output_dir+self.output_name
        if not os.path.exists(test_dir_name):
            os.mkdir(test_dir_name)
        test_dir_name=test_dir_name+"/"+seq_name
        if not os.path.exists(test_dir_name):
            os.mkdir(test_dir_name)

        expstep = 0
        for f, img_file in enumerate(img_files):
            expstep += 1
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box, test_dir_name=test_dir_name, expstep=str(expstep).zfill(6), visualize=visualize)
            else:
                boxes[f, :], center, track_center, mean_center, max_val, track_max, norm, beta, running_mean_delta = self.update(img, test_dir_name=test_dir_name, expstep=str(expstep).zfill(6), visualize=visualize)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :], expstep=str(expstep).zfill(6), dir=test_dir_name, center=center, track_center=track_center, mean_center=mean_center, max_val=max_val, track_max=track_max, norm=norm, beta=beta, running_mean_delta=running_mean_delta)

        return boxes, times


    #entrainement de l'extracteur (self.net) avec l'optimiseur self.criterion (BalancedLoss)
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    #étpae de valisation (qui n'existait pas du tout dans la version forkée)
    @torch.no_grad()
    def val_step (self, batch):
        self.net.eval()
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        responses = self.net(z, x)
        labels = self._create_labels(responses.size())
        loss = self.criterion(responses, labels, viz=True)
        return loss

    #itération sur le dataset pour validation
    @torch.no_grad()
    def val_over(self, val_dataloader):
        mean = []
        print("Computing val loss...")
        for it, batch in enumerate(val_dataloader) :
            loss = self.val_step(batch).item()
            #print(f'Val [{it + 1}/{len(val_dataloader)}] BalancedLoss() : {round(loss, 10)}')
            mean.append(loss)
        print("ValLoss : ", np.mean(np.array(mean)))
        return float(np.mean(np.array(mean)))
            
    #itération sur le dataset pour l'entrainement
    @torch.enable_grad()
    def train_over(self, seqs, val_seqs, save_dir='pretrained'):

        now = datetime.datetime.now()
        dir_name = "[TRAIN]siamfc-"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"-"+str(now.hour)+"_"+str(now.minute)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        self.net.train()
        save_dir = dir_name

        # setup dataset (train et val)
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        val_dataset = Pair(
            seqs=val_seqs,
            transforms=transforms)
        
        # setup dataloader (train et val)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        #enregistre une trace des paramètres de l'experience dans le dossier
        with open(dir_name+f'/params.json', 'w') as file :
            json.dump(self.params_summary,file)

        #graphs
        X = []
        Lloss = []
        Lval = []
        Llabels = []
        history = {}
        
        step = 0
        t0 = time.time()

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            history[epoch+1] = {}

            # loop over dataloader
            for it, batch in enumerate(dataloader) :

                step+=1
                if it != 0 : Llabels.append("")
                X.append(step)

                step += 1
                loss = self.train_step(batch, backward=True)
                Lloss.append(loss)
                
                t = time.time() - t0

                print(f'Epoch: {epoch + 1} [{it + 1}/{len(dataloader)}] ({round(t, 2)}s) {str(self.criterion)}: {round(loss, 10)}')
                sys.stdout.flush()
                
                #validation tous les 100 batch
                if (it+1)%100 == 0 :
                    valoss = self.val_over(val_dataloader)
                else :
                    valoss = None
                Lval.append(valoss)

                history[epoch+1][it+1] = {"it":it+1, "tot":len(dataloader), "time": t, str(self.criterion) : loss, "val" : valoss}
                
            #affichage des courbes pendant l'apprentissage

            Llabels.append(f"ep. {epoch} - {round(time.time() - t0, 2)}sec")

            fig, ax = plt.subplots()
            fig.canvas.draw()

            plt.plot(X, Lloss, label = {str(self.criterion)})
            #calcul de la moyenne glissante pour lisser les courbes
            Lloss_mean = list(np.convolve(Lloss, np.ones(40)/40, mode='valid'))
            plt.plot(X, Lloss_mean+[None for _ in range(len(X)-len(Lloss_mean))], label = {str(self.criterion)+" (moving average)"})
            
            plt.legend()

            fig.canvas.draw()

            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels[-1] = f"ep. {epoch+1} - {round(time.time() - t0, 2)}sec"
            ax.set_xticklabels(labels)

            #enregistrement de l'historique d'apprentissage (pour les courbes)
            with open(dir_name+f'/train_ep{epoch+1}.json', 'w') as file :
                json.dump(history,file)
            
            plt.savefig(dir_name+f'/siamfc_loss_ep{epoch+1}.png')

            plt.clf()
            plt.cla()
            plt.close()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    

    #fonction de création des cartes "objectifs" (cf. "méthode d'apprentissage de l'extracteur" dans le rapport)
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
