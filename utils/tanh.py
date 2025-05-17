import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
steepness_values = [1,2,3,4,5,6,7]

plt.figure(figsize=(8, 6))

for s in steepness_values:
    y = np.tanh(s * x)
    plt.plot(x, y, label=f'tanh({s}x)')

plt.title('Tanh Functions with Different Steepness')
plt.xlabel('x')
plt.ylabel('tanh(s * x)')
plt.legend()
plt.grid(True)

plt.savefig('/home/congshan/uav/uav_roundup/navigation_strategy/utils/tanh_steepness_variation.png')
