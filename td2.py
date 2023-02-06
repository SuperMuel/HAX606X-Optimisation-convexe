import numpy as np
import matplotlib.pyplot as plt

# %% Exercice 4


def f(x, y, a):
    return x**2 + y**2 + a*x*y - 2*x - 2 * y


x = np.linspace(-5, 5)
y = np.linspace(-5, 5)

X, Y = np.meshgrid(x, y)
fig = plt.figure()


ax = fig.add_subplot(1, 2, 1, projection="3d")
Z = f(X, Y, 0)
ax.plot_surface(X, Y, Z)

ax = fig.add_subplot(1, 2, 2, projection="3d")
Z = f(X, Y, 1)
ax.plot_surface(X, Y, Z)


plt.tight_layout()
plt.show()

# %%
