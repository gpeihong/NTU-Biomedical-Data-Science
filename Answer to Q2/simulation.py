from tqdm import tqdm
import matplotlib.pyplot as plt

Ce=1      #the initial concentration
Cs=10
Ces=0
Cp=0

k1,k2,k3=100,600,150
dt=1e-5          #Set the size of the unit time
total_e = []     #List of substance concentration at each time
total_s = []
total_es = []
total_p = []
for i in tqdm(range(100000)):
    
    CE, CS, CES, CP = Ce, Cs, Ces, Cp
    
    CE = CE - k1*CE*CS*dt + k2*CES*dt + k3*CES*dt
    CS = CS - k1*CE*CS*dt + k2*CES*dt
    CES = CES + k1*CE*CS*dt - k3*CES*dt - k2*CES*dt
    CP = CP + k3*CES*dt

    if CS<=1e-6:
        break
    
    Ce, Cs, Ces, Cp = CE, CS, CES, CP

    total_e.append(Ce)
    total_s.append(Cs)
    total_es.append(Ces)
    total_p.append(Cp)


plt.plot(total_e,color='deepskyblue')
plt.plot(total_s)
plt.plot(total_es)
plt.plot(total_p,color='blueviolet')
plt.legend(["Ce","Cs","Ces","Cp"])
plt.show()   #Show a general trend chart