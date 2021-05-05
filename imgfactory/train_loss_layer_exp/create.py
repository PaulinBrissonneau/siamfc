import json
import numpy as np
from matplotlib import pyplot as plt



with open("imgfactory/train_loss_layer_exp/train_ep60_8.json", 'r') as file :
    d1 = json.load(file)

def convolve (L1) :
    L1_av = list(np.convolve(L1, np.ones(500)/500, mode='valid'))
    L1_av = L1_av+[None for _ in range(len(L1)-len(L1_av))]
    return L1_av

Lx = []
L1 = []
L2 = []

i = 0
for ep in d1.keys() :

    epi = d1[ep]

    for k, v in epi.items() :
        i+=1
        Lx.append(i)
        L1.append(epi[k]['BalancedLoss()'])
        L2.append(epi[k]['val'])

L1_av = convolve (L1)

#print(L2)

plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(5.5,2.75))

ax.locator_params(axis='y', nbins=10)
ax.locator_params(axis='x', nbins=1)

ax.set_ylim([0.2,0.8]) #0.2 0.7

ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.xlabel("Itérations (batch)")
plt.ylabel("Coût (BalancedLoss)")

Lx_sparse = []
L1_av_sparse = []
L2_sparse = []

k = 0
for i in range(len(Lx)) :
    Lx_sparse.append(Lx[i])
    L1_av_sparse.append(L1_av[i])
    if L2[i] != None :
        k += 1
        #print(k)
        if k%5 == 0 :
            L2_sparse.append(L2[i])
        else :
            L2_sparse.append(None)
    else :
        L2_sparse.append(None)

#print(Lx_sparse)
#print(L1_av_sparse)
#print(L2_sparse)

plt.plot(Lx_sparse, L1_av_sparse, label="train", color='grey', zorder=0)
plt.scatter(Lx_sparse, L2_sparse, marker='^', label="val", color='black', s=5, zorder=10)

plt.axhline(y=0.693, color='black', linestyle='--')

plt.legend()
plt.show()
plt.savefig("imgfactory/train_loss_layer_exp/8layers.png")