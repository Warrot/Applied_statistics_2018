import numpy as np
import matplotlib.pyplot as plt
import pprint

print('')

t = []
vol = []

for i in range(2):
    t.append(np.loadtxt('defaultboi'+str(i)+'.csv', dtype='float', delimiter=',', skiprows=11, unpack=True)[0])
    vol.append(np.loadtxt('defaultboi'+str(i)+'.csv', dtype='float', delimiter=',', skiprows=11, unpack=True)[1])
 
    #bob = np.concatenate((t, time), axis=1)
    #bobs = np.concatenate((vol, voltage), axis=1)
    
    
pp = pprint.PrettyPrinter(depth=10)
pp.pprint(t)
pp.pprint(vol)
#    for k in range(len(time)):
#        np.add(t[i], time[k])




#plt.plot(t,vol)