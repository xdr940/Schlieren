import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import masked_array
import matplotlib.cm as cm
v1= plt.imread('./pics/')
v1 = -1+2*np.random.rand(50,150)
v1a = masked_array(v1,v1<0)
v1b = masked_array(v1,v1>=0)
fig,ax = plt.subplots()
pa = ax.imshow(v1a,interpolation='nearest',cmap=cm.Reds)
cba = plt.colorbar(pa,shrink=0.25)
pb = ax.imshow(v1b,interpolation='nearest',cmap=cm.winter)
cbb = plt.colorbar(pb,shrink=0.25)
plt.xlabel('Day')
plt.ylabel('Depth')
cba.set_label('positive')
cbb.set_label('negative')
plt.show()
