# %%
# Début de cellule
print(1+3)  # commentaire en ligne
# %%
# Une autre cellule
print(2**3)  # commentaire en ligne

# %%
import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = np.cos(2 * np.pi * r)

fig, ax = plt.subplots()
ax.plot(r,theta)
ax.grid(True)
plt.show()
# %%
