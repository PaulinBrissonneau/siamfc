import json
import numpy as np
from matplotlib import pyplot as plt

with open("imgfactory/test_perf_layer/performance_CNN2.json", 'r') as file :
    p2 = json.load(file)
with open("imgfactory/test_perf_layer/performance_CNN3.json", 'r') as file :
    p3 = json.load(file)
with open("imgfactory/test_perf_layer/performance_CNN4.json", 'r') as file :
    p4 = json.load(file)
with open("imgfactory/test_perf_layer/performance_CNN5.json", 'r') as file :
    p5 = json.load(file)
with open("imgfactory/test_perf_layer/performance_CNN6.json", 'r') as file :
    p6 = json.load(file)
with open("imgfactory/test_perf_layer/performance_CNN7.json", 'r') as file :
    p7 = json.load(file)
with open("imgfactory/test_perf_layer/performance_CNN8.json", 'r') as file :
    p8 = json.load(file)

o1 = p2["SiamFC"]["overall"]
o2 = p3["SiamFC"]["overall"]
o3 = p4["SiamFC"]["overall"]
o4 = p5["SiamFC"]["overall"]
o5 = p6["SiamFC"]["overall"]
o6 = p7["SiamFC"]["overall"]
o7 = p8["SiamFC"]["overall"]

Lref_succ1 = o1['success_curve']
Lref_pres1 = o1['precision_curve']

Lref_succ2 = o2['success_curve']
Lref_pres2 = o2['precision_curve']

Lref_succ3 = o3['success_curve']
Lref_pres3 = o3['precision_curve']

Lref_succ4 = o4['success_curve']
Lref_pres4 = o4['precision_curve']

Lref_succ5 = o5['success_curve']
Lref_pres5 = o5['precision_curve']

Lref_succ6 = o6['success_curve']
Lref_pres6 = o6['precision_curve']

Lref_succ7 = o7['success_curve']
Lref_pres7 = o7['precision_curve']


Lx_pres = list(np.linspace(0, 50, len(Lref_pres1)))
Lx_succ = list(np.linspace(0, 1, len(Lref_succ1)))

plt.style.use(['ieee', 'science', 'grid'])

fig, ax = plt.subplots(figsize=(3,3))


plt.plot(Lx_succ, Lref_succ1, label="2 couches")
plt.plot(Lx_succ, Lref_succ2, label="3 couches")
plt.plot(Lx_succ, Lref_succ3, label="4 couches")
plt.plot(Lx_succ, Lref_succ4, label="5 couches")
plt.plot(Lx_succ, Lref_succ5, label="6 couches")
plt.plot(Lx_succ, Lref_succ6, label="7 couches")
#plt.plot(Lx_succ, Lref_succ7, label="8 couches")

plt.xlabel("Seuil d'overlap")
plt.ylabel("Success")

plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_succ.png")

plt.clf()

fig, ax = plt.subplots(figsize=(3,3))

plt.plot(Lx_pres, Lref_pres1, label="2 couches")
plt.plot(Lx_pres, Lref_pres2, label="3 couches")
plt.plot(Lx_pres, Lref_pres3, label="4 couches")
plt.plot(Lx_pres, Lref_pres4, label="5 couches")
plt.plot(Lx_pres, Lref_pres5, label="6 couches")
plt.plot(Lx_pres, Lref_pres6, label="7 couches")
#plt.plot(Lx_pres, Lref_pres7, label="8 couches")

plt.xlabel("Seuil d'erreur de localisation")
plt.ylabel("Precision")


plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_prec.png")


L = [o1, o2, o3, o4, o5, o6] #o7
Lsuc = [o['success_score'] for o in L]
Lpres = [o['precision_score'] for o in L]
Lrate = [o['success_rate'] for o in L]
Lfps = [o['speed_fps'] for o in L]


plt.clf()
fig, ax = plt.subplots(figsize=(5,5))
plt.plot([i+2 for i in range(len(L))], Lsuc, color="black", linestyle='--', marker='o')
plt.xlabel("Nombre de couches")
plt.ylabel("Success score")
ax.set_ylim([0,1])
#plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_fctlay_suc.png")

plt.clf()
fig, ax = plt.subplots(figsize=(5,5))
plt.plot([i+2 for i in range(len(L))], Lpres, color="black", linestyle='--', marker='o')
plt.xlabel("Nombre de couches")
plt.ylabel("Precision score")
ax.set_ylim([0,1])
#plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_fctlay_prec.png")

plt.clf()
fig, ax = plt.subplots(figsize=(5,5))
plt.plot([i+2 for i in range(len(L))], Lrate, color="black", linestyle='--', marker='o')
plt.xlabel("Nombre de couches")
plt.ylabel("Success rate")
ax.set_ylim([0,1])
#plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_fctlay_rate.png")

plt.clf()
fig, ax = plt.subplots(figsize=(4,3))
plt.plot([i+2 for i in range(len(L))], Lsuc, color="black", linestyle='--', marker='v', label="AUC")
plt.plot([i+2 for i in range(len(L))], Lpres, color="black", linestyle='--', marker='^', label="Précision")
plt.plot([i+2 for i in range(len(L))], Lrate, color="black", linestyle='--', marker='o', label="Taux de succès")
plt.xlabel("Nombre de couches")
ax.set_ylim([0,1])
plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_fctlay_perfs.png")


plt.clf()
fig, ax = plt.subplots(figsize=(5,5))

plt.plot([i+2 for i in range(len(L))], Lfps, color="black", linestyle='--', marker='o')

plt.xlabel("Nombre de couches")
plt.ylabel("Images par seconde")

#plt.legend()
plt.savefig("imgfactory/test_perf_layer/courbes_fctlay_fps.png")