#Code forké depuis "huanglianghua" (https://github.com/huanglianghua/siamfc-pytorch)
#Adapté et modifié par Paulin Brissonneau


"""
Point d'entrée pour les tests.
Les tests sont effectués par "got10k.experiments" une bibliothèque qui permet de normaliser les exprimentations.
L'avantage de cette librairie est qu'on ne s'occupe pas de gérer les données, le chargement de séquences vidéos est automatique.
Le calcul des performances est aussi automatique.
"""

from __future__ import absolute_import
import os
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':

    siamfc_dir = "/workspace/code/IA/tracking/siamfc"
    epoch = 1
    version = "[TRAIN]siamfc-2021_2_19-16_52"
    net_path = f"{siamfc_dir}/{version}/siamfc_alexnet_e{epoch}.pth"
    output_name = "center_brut"
    tracker = TrackerSiamFC(net_path=net_path, output_name=output_name)

    e = ExperimentOTB('./data/OTB', version=2015)
    e.run(tracker, visualize=True)
    e.report([tracker.name])
