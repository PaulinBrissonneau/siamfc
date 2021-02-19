#Code forké depuis "huanglianghua" (https://github.com/huanglianghua/siamfc-pytorch)
#Adapté et modifié par Paulin Brissonneau

"""
Point d'entrée pour les entrainements des extracteurs.
Se contente d'appeler "TrackerSiamFC" et sa méthode "train_over".
"""

from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC

if __name__ == '__main__':
    root_dir = os.path.expanduser('./data/GOT-10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    seqs_val = GOT10k(root_dir, subset='val', return_meta=True)

    exp = [4] #permet de faire plusieurs apprentissages à la suite en changeant des hyper-paramètres
    for n_layer in exp : 
        tracker = TrackerSiamFC(n_layer=n_layer)
        tracker.train_over(seqs, val_seqs=seqs_val)
