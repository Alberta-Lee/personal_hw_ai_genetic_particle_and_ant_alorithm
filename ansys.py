import numpy as np
import matplotlib.pyplot as plt

gen = np.load('./res/genetic.npy')
par = np.load('./res/particle.npy')
par = par[:-1]
ant = np.load('./res/ant.npy')

I = np.arange(len(gen))

plt.plot(I, gen, color='b', linestyle=':', marker='o', markerfacecolor='r', markersize=1, label='genetic')
plt.plot(I, par, color='g', linestyle='--', marker='*', markerfacecolor='b', markersize=1, label='particle')
plt.plot(I, ant, color='r', linestyle='-', marker='+', markerfacecolor='g', markersize=1, label='ant colony')
plt.title('Result comparison', fontsize=16)
plt.xlabel('number of iterations', fontsize=12)
plt.ylabel('distance', fontsize=12)
plt.legend()
plt.show()