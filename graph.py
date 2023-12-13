from matplotlib import pyplot as plt
import numpy as np

data = np.load('npydata/fittest-avg.npy')
x = np.arange(0, len(data), 1)

plt.clf()
plt.plot(x, data)

plt.xlabel('x - Iterations') 

plt.ylabel('y - Fittness') 
  
# giving a title to my graph 
plt.title('Average Fittness') 

plt.savefig('npydata/Average Fittness')

#second graph
data = np.load('npydata/fittest.npy')
x = np.arange(0, len(data), 1)

plt.clf()
plt.plot(x, data)

plt.xlabel('x - Iterations') 

plt.ylabel('y - Fittness') 
  
# giving a title to my graph 
plt.title('Fittness') 
plt.savefig('npydata/Fittness')