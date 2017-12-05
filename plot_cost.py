import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('comm_depth_1_test.npz')
data2 = np.load('comm_depth_3_test.npz')
data3 = np.load('comm_depth_5_test.npz')
plt.plot(data1['ccost'],label="depth = 1")
plt.plot(data2['ccost'],label="depth = 3")
plt.plot(data3['ccost'],label = "depth = 5")
plt.legend(loc = 0, ncol = 2)
plt.show()
plt.plot(data1['ecost'],label = "depth = 1")
plt.plot(data2['ecost'],label = "depth = 3")
plt.plot(data3['ecost'],label = "depth = 5")
plt.legend(loc = 0, ncol = 2)
plt.show()


data1 = np.load('rand_4_10.npz')
data2 = np.load('rand_2_10.npz')
data3 = np.load('rand_1_10.npz')
plt.plot(data1['ccost'],label="r=4,d=10")
plt.plot(data2['ccost'],label="r=2,d=10")
plt.plot(data3['ccost'],label = "r=1,d=10")
plt.legend(loc = 0, ncol = 2)
plt.show()
plt.plot(data1['ecost'],label = "r=4,d=10")
plt.plot(data2['ecost'],label = "r=2,d=10")
plt.plot(data3['ecost'],label = "r=1,d=10")
plt.legend(loc = 0, ncol = 2)
plt.show()

'''

data1 = np.load('rand_4_5.npz')
data2 = np.load('rand_4_4.npz')
data3 = np.load('rand_4_3.npz')
data4 = np.load('rand_4_2.npz')
plt.plot(data1['ccost'],label="r=4,d=5")
plt.plot(data2['ccost'],label="r=4,d=4")
plt.plot(data3['ccost'],label = "r=4,d=3")
plt.plot(data4['ccost'],label = "r=4,d=2")
plt.legend(loc = 0, ncol = 2)
plt.show()
plt.plot(data1['ecost'],label = "r=4,d=5")
plt.plot(data2['ecost'],label = "r=4,d=4")
plt.plot(data3['ecost'],label = "r=4,d=3")
plt.plot(data4['ecost'],label = "r=4,d=2")
plt.legend(loc = 0, ncol = 2)
plt.show()

'''

data1 = np.load('rand_8_10.npz')
data2 = np.load('rand_8_5.npz')
data3 = np.load('rand_8_2.npz')
data4 = np.load('rand_8_1.npz')
plt.plot(data1['ccost'],label="r=8,d=10")
plt.plot(data2['ccost'],label="r=8,d=5")
plt.plot(data3['ccost'],label = "r=8,d=2")
plt.plot(data4['ccost'],label = "r=8,d=1")
plt.legend(loc = 0, ncol = 2)
plt.show()
plt.plot(data1['ecost'],label = "r=8,d=10")
plt.plot(data2['ecost'],label = "r=8,d=5")
plt.plot(data3['ecost'],label = "r=8,d=2")
plt.plot(data4['ecost'],label = "r=8,d=1")
plt.legend(loc = 0, ncol = 2)
plt.show()
'''


data1 = np.load('rand_4_5.npz')
data2 = np.load('rand_8_5.npz')

plt.plot(data1['ccost'],label="r=4,d=5")
plt.plot(data2['ccost'],label="r=8,d=5")

plt.legend(loc = 0, ncol = 2)
plt.show()
plt.plot(data1['ecost'],label = "r=4,d=5")
plt.plot(data2['ecost'],label = "r=8,d=5")

plt.legend(loc = 0, ncol = 2)
plt.show()
'''