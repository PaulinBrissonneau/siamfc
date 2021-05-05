import json
import numpy as np
from matplotlib import pyplot as plt

with open("imgfactory/test_perf/perf_lambda.json", 'r') as file :
    p1 = json.load(file)
with open("imgfactory/test_perf/perf_V0_sansBN.json", 'r') as file :
    p2 = json.load(file)

o1 = p1["SiamFC"]["overall"]
o2 = p2["SiamFC"]["overall"]

Lref_succ1 = o1['success_curve']
Lref_pres1 = o1['precision_curve']
Lref_succ2 = o2['success_curve']
Lref_pres2 = o2['precision_curve']


Lx_pres = list(np.linspace(0, 50, len(Lref_pres1)))
Lx_succ = list(np.linspace(0, 1, len(Lref_succ1)))

plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(2.5,2.5))

#ax.locator_params(axis='y', nbins=10)
#ax.locator_params(axis='x', nbins=1)

#ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.plot(Lx_succ, Lref_succ1, label="avec BN", color='black')
plt.plot(Lx_succ, Lref_succ2, '-.', label="sans BN", color='black')
plt.xlabel("Seuil d'overlap")
plt.ylabel("Success")

plt.legend()
plt.savefig("imgfactory/test_perf/courbes_succ.png")

plt.clf()

fig, ax = plt.subplots(figsize=(2.5,2.5))

#ax.locator_params(axis='y', nbins=10)
#ax.locator_params(axis='x', nbins=1)

#ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.plot(Lx_pres, Lref_pres1, label="avec BN", color='black')
plt.plot(Lx_pres, Lref_pres2, '-.', label="sans BN", color='black')
plt.xlabel("Seuil d'erreur de localisation")
plt.ylabel("Precision")


plt.legend()
#plt.show()
plt.savefig("imgfactory/test_perf/courbes_pre.png")