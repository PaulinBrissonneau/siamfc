import json
import numpy as np
from matplotlib import pyplot as plt

with open("imgfactory/test_perf_gamma/performance_center.json", 'r') as file :
    p1 = json.load(file)
with open("imgfactory/test_perf_gamma/performance_beta08_gamma02.json", 'r') as file :
    p2 = json.load(file)

subs = p1["SiamFC"]["seq_wise"].keys()

o1 = p1["SiamFC"]["seq_wise"]["David3"]
o2 = p2["SiamFC"]["seq_wise"]["David3"]
o3 = p1["SiamFC"]["seq_wise"]["ClifBar"]
o4 = p2["SiamFC"]["seq_wise"]["ClifBar"]


Lref_succ1 = o1['success_curve']
Lref_pres1 = o1['precision_curve']

Lref_succ2 = o2['success_curve']
Lref_pres2 = o2['precision_curve']

Lref_succ3 = o3['success_curve']
Lref_pres3 = o3['precision_curve']

Lref_succ4 = o4['success_curve']
Lref_pres4 = o4['precision_curve']

Lx_pres = list(np.linspace(0, 50, len(Lref_pres1)))
Lx_succ = list(np.linspace(0, 1, len(Lref_succ1)))

plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(4,4))

plt.plot(Lx_succ, Lref_succ1, label="référence (seq. David3)",  color='blue')
plt.plot(Lx_succ, Lref_succ2, '-.', label="noyau glissant (seq. David3)",  color='blue')
plt.plot(Lx_succ, Lref_succ3, label="référence (seq. ClifBar)",  color='black')
plt.plot(Lx_succ, Lref_succ4, '-.', label="noyau glissant (seq. ClifBar)",  color='black')


plt.xlabel("Seuil d'overlap")
plt.ylabel("Success")

plt.legend()
plt.savefig("imgfactory/test_perf_gamma/courbes_succ_mutli_bof.png")

plt.clf()

fig, ax = plt.subplots(figsize=(4,4))

plt.plot(Lx_pres, Lref_pres1, label="référence (seq. David3)",  color='blue')
plt.plot(Lx_pres, Lref_pres2, '-.', label="noyau glissant (seq. David3)",  color='blue')
plt.plot(Lx_pres, Lref_pres3, label="référence (seq. ClifBar)",  color='black')
plt.plot(Lx_pres, Lref_pres4, '-.', label="noyau glissant (seq. ClifBar)",  color='black')

plt.xlabel("Seuil d'erreur de localisation")
plt.ylabel("Precision")

plt.legend()
plt.savefig("imgfactory/test_perf_gamma/courbes_prec_multi_bof.png")




"""

"""