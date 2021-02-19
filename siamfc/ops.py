#Code forké depuis "huanglianghua" (https://github.com/huanglianghua/siamfc-pytorch)
#Adapté et modifié par Paulin Brissonneau

"""
Regroupe plusieurs opérations élémentaires utiles dans plusieurs parties du programme.
Beaucoup d'ajouts sur la visualisation des données internes du système siamois.
"""


from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import os


def init_weights_module(m, gain=1) :
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_array(x, label, dir=None, from_np=False, norm=True, expstep=None, marker=None):

    if dir is not None :
        dir_name = f"{dir}/{label}"
    else :
        dir_name = f"imgfactory/{label}"


    if not from_np : x = x.cpu().detach().numpy()
    if norm : x = x*255
    im1 = Image.fromarray(x.astype(np.uint8))

    if marker is not None :
        draw = ImageDraw.Draw(im1)
        draw.ellipse((marker[1], marker[0], marker[1]+10, marker[0]+10), fill='black')

    if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    
    im1.save(f"{dir_name}/{expstep}.png")
        

def show_image(img, boxes=None, dir=None, expstep=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=False,
               cvt_code=cv2.COLOR_RGB2BGR, center=None, track_center=None, mean_center=None, max_val=None, track_max=None, norm=None, beta=None, running_mean_delta=None):

    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    dir_name = f"{dir}/video"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)

        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

        if center is not None:
            center = np.array(center, dtype=np.float32) * scale
            
        if track_center is not None:
            track_center = np.array(track_center, dtype=np.float32) * scale
            
        if mean_center is not None:
            mean_center = np.array(mean_center, dtype=np.float32) * scale 
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    #juste pour générer annexe rapport
    center = None
    track_center = None
    mean_center = None

    if center is not None:
        center = np.array(center, dtype=np.int32)
        img = cv2.circle(img, (center[1], center[0]), 2, color=(0,0,255),  thickness=2) #rouge

    if track_center is not None:
        track_center = np.array(track_center, dtype=np.int32)
        img = cv2.circle(img, (track_center[1], track_center[0]), 5, color=(0,255,0),  thickness=2) #vert

    if mean_center is not None:
        mean_center = np.array(mean_center, dtype=np.int32)
        img = cv2.circle(img, (mean_center[1], mean_center[0]), 2, color=(255,0,0),  thickness=2) #bleu
    
    cv2.imwrite(f"{dir_name}/{expstep}.png", img)

    return img


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch
