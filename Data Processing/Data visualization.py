import matplotlib.pyplot as plt
import numpy as np

heartbeat1 = np.load('data_1.npy')
heartbeat2 = np.load('data_2.npy')
plt.plot(heartbeat1)
plt.plot(heartbeat2)
plt.show()