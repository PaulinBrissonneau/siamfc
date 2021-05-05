import json
import numpy as np
from matplotlib import pyplot as plt



with open("imgfactory/train_loss_tasks/train_ep2_cla.json", 'r') as file :
    d1 = json.load(file)
with open("imgfactory/train_loss_tasks/train_ep2_seg.json", 'r') as file :
    d2 = json.load(file)
with open("imgfactory/train_loss_tasks/train_ep2_cla_lay4.json", 'r') as file :
    d3 = json.load(file)
with open("imgfactory/train_loss_tasks/train_ep2_seg_lay4.json", 'r') as file :
    d4 = json.load(file)

def convolve (L1) :
    L1_av = list(np.convolve(L1, np.ones(40)/40, mode='valid'))
    L1_av = L1_av+[None for _ in range(len(L1)-len(L1_av))]
    return L1_av

ep1_1 = d1["1"]
ep1_2 = d2["1"]
ep1_3 = d3["1"]
ep1_4 = d4["1"]
ep2_1 = d1["2"]
ep2_2 = d2["2"]
ep2_3 = d3["2"]
ep2_4 = d4["2"]


Lx = []
L1 = []
L2 = []
Lval2 = []
L3 = []
L4 = []

last=0
for k, v in ep1_1.items() :
    Lx.append(k)
    L1.append(ep1_1[k]['BalancedLoss()'])
    L2.append(ep1_2[k]['BalancedLoss()'])
    Lval2.append(ep1_2[k]['val'])
    L3.append(ep1_3[k]['BalancedLoss()'])
    L4.append(ep1_4[k]['BalancedLoss()'])
    last=k

for k, v in ep2_1.items() :
    Lx.append(last+k)
    L1.append(ep2_1[k]['BalancedLoss()'])
    L2.append(ep2_2[k]['BalancedLoss()'])
    Lval2.append(ep2_2[k]['val'])
    L3.append(ep2_3[k]['BalancedLoss()'])
    L4.append(ep2_4[k]['BalancedLoss()'])
  
L1_av = convolve (L1)
L2_av = convolve (L2)
L3_av = convolve (L3)
L4_av = convolve (L4)

for i in range(len(L4_av)) :
    if L4_av[i] != None and L4_av[i] > 1 : L4_av[i] = None


for i in range(len(Lval2)) :
    if Lval2[i] != None and Lval2[i] > 1 : Lval2[i] = None


plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(5.5,2.75))

ax.locator_params(axis='y', nbins=10)
ax.locator_params(axis='x', nbins=1)

ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.xlabel("Itérations (batch)")
plt.ylabel("Coût (BalancedLoss)")

plt.plot(Lx, L1_av, '-', label="classification", color='black')
plt.plot(Lx, L2_av, '-.', label="segmentation", color='black')
plt.scatter(Lx, Lval2, marker='^', label="segmentation val loss", color='black', s=5)
plt.plot(Lx, L3_av, '-', label="classification/layer4", color='red')
plt.plot(Lx, L4_av, '-.', label="segmentation/layer4", color='red')

plt.legend()
plt.show()
plt.savefig("imgfactory/train_loss_tasks/courbes.png")