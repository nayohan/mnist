import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



x = np.arange(0,6,0.1)
y = np.sin(x)
print(x)
print(y)

plt.plot(x, y)
plt.show(block=True)
matplotlib.pyplot.show()
print("hello")
