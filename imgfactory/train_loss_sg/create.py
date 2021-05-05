import json
import numpy as np
from matplotlib import pyplot as plt



with open("imgfactory/train_loss_sg/train_ep100_top_train.json", 'r') as file :
    d1 = json.load(file)

"""
mini = 1000
epmin = 0
kmin = 0

for ep in d1.keys() :

    epi = d1[ep]

    for k, v in epi.items() :

        val = epi[k]['val']

        if val is not None and val < mini and k==1000 :
            mini = val
            epmin = ep
            kmin = k
"""

    
print(mini)
print(ep)
print(kmin)



def convolve (L1) :
    L1_av = list(np.convolve(L1, np.ones(500)/500, mode='valid'))
    L1_av = list(np.convolve(L1, np.ones(50)/50, mode='valid'))
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



plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(5.5,2.75))

ax.locator_params(axis='y', nbins=10)
ax.locator_params(axis='x', nbins=1)

ax.set_ylim([0,0.7]) #0.2 0.7

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
        if k%50 == 0 :
            L2_sparse.append(L2[i])
        else :
            L2_sparse.append(None)
    else :
        L2_sparse.append(None)

plt.plot(Lx_sparse, L1_av_sparse, label="train", color='grey', zorder=0)
plt.scatter(Lx_sparse, L2_sparse, marker='^', label="val", color='black', s=5, zorder=10)


#print(Lx_sparse)
#print(L1_av_sparse)
#print(L2_sparse)

#plt.plot(Lx, L1_av, label="train", color='grey', zorder=0)
#plt.scatter(Lx, L2, marker='^', label="val", color='black', s=5, zorder=10)



plt.legend()
plt.show()
plt.savefig("imgfactory/train_loss_sg/courbes_toptrain.png")