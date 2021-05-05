import json
import numpy as np
from matplotlib import pyplot as plt



with open("imgfactory/train_loss_comp/train_ep2_classpt.json", 'r') as file :
    d1 = json.load(file)
with open("imgfactory/train_loss_comp/train_ep2_pasclasspt.json", 'r') as file :
    d2 = json.load(file)
with open("imgfactory/train_loss_comp/train_ep2_lay4.json", 'r') as file :
    d3 = json.load(file)
with open("imgfactory/train_loss_comp/train_ep2_alex.json", 'r') as file :
    d4 = json.load(file)
with open("imgfactory/train_loss_comp/train_ep2_lay3.json", 'r') as file :
    d5 = json.load(file)
with open("imgfactory/train_loss_comp/train_ep2_lay2.json", 'r') as file :
    d6 = json.load(file)
with open("imgfactory/train_loss_comp/train_ep2_lay1.json", 'r') as file :
    d7 = json.load(file)

def convolve (L1) :
    L1_av = list(np.convolve(L1, np.ones(40)/40, mode='valid'))
    L1_av = L1_av+[None for _ in range(len(L1)-len(L1_av))]
    return L1_av

ep1_1 = d1["1"]
ep1_2 = d2["1"]
ep1_3 = d3["1"]
ep1_4 = d4["1"]
ep1_5 = d5["1"]
ep1_6 = d6["1"]
ep1_7 = d7["1"]

Lx = []
L1 = []
L2 = []
L3 = []
L4 = []
L5 = []
L6 = []
L7 = []

for k, v in ep1_1.items() :
    Lx.append(k)
    L1.append(ep1_1[k]['BalancedLoss()'])
    L2.append(ep1_2[k]['BalancedLoss()'])
    L3.append(ep1_3[k]['BalancedLoss()'])
    L4.append(ep1_4[k]['BalancedLoss()'])
    L5.append(ep1_5[k]['BalancedLoss()'])
    L6.append(ep1_6[k]['BalancedLoss()'])
    L7.append(ep1_7[k]['BalancedLoss()'])

L1_av = convolve (L1)
L2_av = convolve (L2)
L3_av = convolve (L3)
L4_av = convolve (L4)
L5_av = convolve (L5)
L6_av = convolve (L6)
L7_av = convolve (L7)

for i in range(len(L3_av)) :
    if L3_av[i] != None and L3_av[i] > 2 : L3_av[i] = None

for i in range(len(L5_av)) :
    if L5_av[i] != None and L5_av[i] > 2 : L5_av[i] = None

for i in range(len(L6_av)) :
    if L6_av[i] != None and L6_av[i] > 2 : L6_av[i] = None

for i in range(len(L7_av)) :
    if L7_av[i] != None and L7_av[i] > 2 : L7_av[i] = None

plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(5.5,2.75))

ax.locator_params(axis='y', nbins=10)
ax.locator_params(axis='x', nbins=1)

ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.xlabel("Itérations (batch)")
plt.ylabel("Coût (BalancedLoss)")

plt.plot(Lx, L1_av, label="resnet", color='#FF8C00')
plt.plot(Lx, L2_av, '-.', label="resnet/classifier", color='#FF4500')
plt.plot(Lx, L3_av, label="resnet/classifier+layer4", color='#FF3300')
plt.plot(Lx, L5_av, label="resnet/classifier+layer3", color='#BB3355')
plt.plot(Lx, L6_av, label="resnet/classifier+layer2", color='#7733AA')
plt.plot(Lx, L7_av, label="resnet/classifier+layer1", color='#3333FF')
plt.plot(Lx, L4_av, '-.', label="alexnet (référence)", color='black')

plt.legend()
plt.show()
plt.savefig("imgfactory/train_loss_comp/courbes.png")