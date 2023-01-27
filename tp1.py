import numpy as np
import matplotlib.pyplot as plt

print(f'La précision des floattants est : {np.finfo(float)}')

x_max = np.finfo(float).max
print(f"{x_max=}")
print(f'{x_max+1=}')

print(f'{x_max == x_max+1 = }')

eps = np.finfo(float).eps

#%%

# Cause une erreur d'overflow : 
x_max == x_max * (1 + eps)

#%% 
# Cause une erreur
1.1 * x_max - 1.1*x_max
# %%

# Tests d'égalité de floattants avec numpy
# assert_almost_equal raises an assertion error if not verified
np.testing.assert_almost_equal(x_max,x_max +1 )
print('test passé')


# %%
i = 1
while 2**(-i) != 0 :
    i +=1
2**(-1074), 2**(-1075)
# %%
# Commence = base**start et finit à base**end et donne [num] valeurs
n_petit = 1200
n_petits= np.logspace(-n_petit, 0,num=n_petit+1, base=2 )

for idx, val in enumerate(n_petits[::-1]):
    print(idx, val)
    if val <= val/2:
        break

np.finfo(np.float64).tiny

# %%

# isclose

np.isclose(0.6, 0.1 + 0.2 + 0.3, rtol=0, atol= 0)
np.isclose(0.6, 0.1 + 0.2 + 0.3, rtol=1, atol= 0)


# %%

A = np.array([1.0, 2, 3])
B = np.array([-1, -2, -3.0])

C = A+B
print( C)


# %%
np.testing.assert_almost_equal(0, C)
# %%
A*B
np.testing.assert_almost_equal(np.array([-1.0, -4, -9]), A*B)
# %%
J = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
from numpy.linalg import matrix_power
I3 = np.eye(3)
np.testing.assert_almost_equal(matrix_power(J,3), I3)


# %%
# Inversion
J_inv = np.linalg.inv(J)
J@J_inv

import time 

# %%
n = 2000
Jbig = np.roll(np.eye(n), -1 , axis = 1)
b = np.arange(n)


t0 = time.perf_counter()
y1 = np.linalg.inv(Jbig)@b
print(y1)
timing_naive = time.perf_counter() - t0
print(f"{timing_naive = }s")
# %%
t1 = time.perf_counter()
y2 = np.linalg.solve(Jbig, b)
timing_optimized = time.perf_counter()-t1

print(f"{timing_optimized = }s")


# %%
# Découpages
# Mettre à zéro une ligne sur 2 de la matrice identité de taille 5\times 55×5

I5 = np.eye(5)
I5[::2,:] = 0
I5

# %%
x = np.linspace(-5,5,100)
x
# %%
y = np.logspace(1,9,base=10,num=9)
y
# %% dimensions
d = np.arange(6)
d.reshape(1,6), d.reshape(6,1)
d.reshape(1,6).shape, d.reshape(6,1).shape

# %%
