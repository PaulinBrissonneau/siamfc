# Projet deep learning 3A CentraleSupélec

Paulin Brissonneau

La plupart des scripts sont forkés depuis https://github.com/huanglianghua/siamfc-pytorch, puis modifiés, adaptés, voire réécrits.

## Contexte

- Le projet *Amélioration du noyau des systèmes siamois pour le suivi d’objets* consiste à implémenter un réseau siamois pour le suivi d'objet et d'étudier une modification du noyau du système. Le système siamois de référence est celui de L. Bertinetto et al. [Fully-convolutional siamese networks for object tracking, 2016](https://arxiv.org/pdf/1606.09549.pdf).
- Le rapport associé au projet explicite les modifications apportées au système de référence. [Il est disponible ici](https://github.com/PaulinBrissonneau/siamfc/blob/main/rapport/rapport.pdf).
- Pour tout problème pour lancer l'algorithme lors de la notation du projet : paulin.brissonneau@student-cs.fr.

## Installation des librairies

##### Les librairies utilisées par les programmes du projet sont :
- Les classiques : matplotlib, PIL, openCV
- Le *framework* `GOT-10k`qui permet de tester les suiveurs : `pip install got10k`
- **pytorch** avec **torchvision** et **cuda** : `conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`

## Jeux de données

Le projet se base sur le *framework* `GOT-10k` qui peut (en théorie) s'adapter à n'importe quelle banque de vidéos de tracking. Cependant le projet a été développé avec le dataset GOT-10k (indépendant du *framework* du même nom) pour l'entrainement et la validation, et OTB pour les tests. Je conseille de reprendre ces deux banques, le bon fonctionnement de la préparation des données est assuré.

> Note : Cela fonctionne aussi très bien de prendre GOT-10k pour les tests (puisque la banque est déjà séparée en train/val/test). Le problème est que les *ground truth* de la séquence de test ne sont pas publiques. Il faut passer par un serveur qui calcul les performances en ligne. Pour avoir testé, ça marche, mais ce n'est pas très pratique, c'est pour ça qu'on utilise OTB en test (comme le conseil aussi [huanglianghua](https://github.com/huanglianghua/siamfc-pytorch) dans le repo du code de référence).

Ces datasets sont disponibles ici :
- [GOT-10k](got-10k.aitestunion.com)
- [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)

Cependant, *framework* `GOT-10k` gère directement le téléchargement des données au bon format. Lors du lancement d'un entrainement, il téléchargera les données au bon format dans `./data/GOT-10k` et `./data/OTB` à la racine du projet.
**Attention.** Ce sont des banques de vidéos de grande résolution, l'espace disque nécessaire est donc assez gros (6Go pour OTB, 70Go pour GOT-10k). Un lien symbolique vers un disque HDD peut être utile mais ralentira les calculs (le chargement des données prend un temps non négligeable lors des calculs). 

> Note : Au moment où j'écris le *readme* (19 fev 2021), le serveur du site de [GOT-10k](got-10k.aitestunion.com) ne répond plus, mais les données sont toujours accessibles en passant par le téléchargement automatique via le *framework* `GOT-10k.

## Lancement d'un entrainement

- Pour lancer un entrainement : `python train.py` après avoir chargé le bon environnement virtuel.
- Cela va créer un nouveau dossier `[TRAIN]siamfc-<date>` à la racine du projet. Ce dossier comportement toutes les informations de l'entrainement : les valeurs des indicateurs en cours d'entrainement et de validation (loss, temps, etc) ; les valeurs fixes de certains hyper-paramètres pour reconnaitre les entrainements ; les checkpoints de sauvegarde des modèles ; etc.

## Lancement d'un test

- Pour lancer une batterie de test : `python test.py` après avoir chargé le bon environnement virtuel.
- Pour choisir quel modèle on teste, on change les valeurs de `test.py`, en particulier, **version** et **epoch**, qui sont respectivement le nom du dossier `[TRAIN]siamfc-<date>` de l'entrainement, et l'epoch du modèle que l'on teste. Le dossier `[TRAIN]siamfc-<date>` est placé tel qu'il a été créé pour l'entrainement, c'est-à-dire à la racine du projet.
- Les résultats de tracking sont enregistrés dans `./results`, puis les performances (*precision*, *success rate*, etc) sont enregistrées dans `./reports`. Ces dossier sont créés automatiquement.
- Par défaut, les résultats qualitatifs des tests (visualisation des noyaux, du suivi, etc), pour chaque séquence, seront enregistrés dans `./outputs/siamfc`. **Attention.** Comme ces résultats re-créent les vidéos, ils sont très lourds. Un lien symbolique vers un HDD peut être nécessaire. On peut se passer de ces résultats visuels en configurant `visualize=False` dans `test.py`.
