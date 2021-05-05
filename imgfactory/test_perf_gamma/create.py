import json
import numpy as np
from matplotlib import pyplot as plt

with open("imgfactory/test_perf_gamma/performance_center.json", 'r') as file :
    p1 = json.load(file)
with open("imgfactory/test_perf_gamma/performance_beta08_gamma02.json", 'r') as file :
    p2 = json.load(file)

subs = p1["SiamFC"]["seq_wise"].keys()

for sub in subs :
    sub1 = p1["SiamFC"]["seq_wise"][sub]["precision_score"]
    sub2 = p2["SiamFC"]["seq_wise"][sub]["precision_score"]
    if sub2 > sub1 :
        print(f"+++++++++{sub}+++++++++++")
        print(str(sub1) +'-'+ str(sub2))
    if sub2 < sub1 :
        print(f"---------{sub}------------")
        print(str(sub1) +'-'+ str(sub2))


#o1 = p1["SiamFC"]["seq_wise"]["Girl2"]
#o2 = p2["SiamFC"]["seq_wise"]["Girl2"]

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

plt.plot(Lx_succ, Lref_succ1, label="référence", color='grey')
plt.plot(Lx_succ, Lref_succ2, '-.', label="noyau glissant", color='black')
plt.xlabel("Seuil d'overlap")
plt.ylabel("Success")

plt.legend()
plt.savefig("imgfactory/test_perf_gamma/courbes_succ.png")

plt.clf()

fig, ax = plt.subplots(figsize=(2.5,2.5))

#ax.locator_params(axis='y', nbins=10)
#ax.locator_params(axis='x', nbins=1)

#ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.plot(Lx_pres, Lref_pres1, label="référence", color='grey')
plt.plot(Lx_pres, Lref_pres2, '-.', label="noyau glissant", color='black')
plt.xlabel("Seuil d'erreur de localisation")
plt.ylabel("Precision")


plt.legend()
#plt.show()
plt.savefig("imgfactory/test_perf_gamma/courbes_pre.png")