import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 3, 5)
y = [1-v if 1-v>0 else 0 for v in x]
plt.plot(x, y)
plt.xscale