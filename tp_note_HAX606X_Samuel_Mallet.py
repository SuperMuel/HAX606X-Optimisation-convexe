# %% [markdown]
# # HAX606X - TP NOTE 1 - MALLET Samuel

# %%
from math import sqrt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from scipy.optimize import rosen_der
import numpy as np
import matplotlib.pyplot as plt

# %%
# Question 1a)


def grad_desc(df, x_init, gamma, eps, maxiter):
    """Descente de gradient normale.
    On pourra utiliser des types python natifs ou des np.ndarray 
    pour x_init et le type d'argument de df.
    """
    x = x_init
    iters = []
    is_numpy = isinstance(x_init, np.ndarray)
    for i in range(maxiter):
        g = df(x)
        norm = np.linalg.norm(g, ord=np.inf) if is_numpy else abs(g)
        if norm <= eps:
            break
        iters.append(x.copy() if is_numpy else x)
        x = x-gamma*g
    return np.array(iters)

# %%
# Question 1b) Test avec un espace de départ de dimension 23


# On définit une matrice symétrique définie positive (très simple) de taille 23x23 :
A = np.eye(23, 23)


def f_23(x):
    return x.T @ A @ x / 2


def f_23_grad(x):
    return A @ x


x_init = np.full(23, 1)

iters = grad_desc(f_23_grad, x_init, gamma=0.3, eps=10e-5, maxiter=500)

print(f"L'algo converge en {len(iters)} itérations.")
print("On peut observer, par exemple, la convergence d'une coordonée des itérés : ")
print(iters[:, 0])


# %%
# Question 1c: Test unidimensionnel

def f(x):
    return (x-1)**2


def df(x):
    return 2*x-2

# Execution de la descente de gradient


x_init = -1/2
iters = grad_desc(df, x_init=x_init, gamma=0.1, eps=10e-15, maxiter=15)

# Graphique

x = np.linspace(-0.5, 1.25, 1000)
y = f(x)

fig, (ax_f, ax_grad) = plt.subplots(2, 1, sharex=True)

# Graphe de la fonction f
ax_f.plot(x, y, label="$f:x\mapsto(x-1)^2$")
ax_f.scatter(iters, f(iters), color="red", label="$f(x_k)$")
ax_f.set_ylabel("$f(x)$")
ax_f.legend()

# Graphe des amplitudes des dérivées en chaque point
ax_grad.scatter(iters, np.abs(df(iters)),
                label=r"$|\nabla f(x_k)|$", marker='d')
ax_grad.set_ylabel(r"|$\nabla f(x)|$")
ax_grad.legend()


# %% [markdown]
# On observe une décroissante linéaire de la norme du gradient.

# %% [markdown]
# # Question 2 : Fonction de Rosenbrock

# %%


def f_ros(x1, x2):
    return (x1-1)**2 + 100 * (x2-x1**2)**2


x_init = np.array([-1, -0.5])
eps = 10e-10

# %%
iters5 = grad_desc(rosen_der, x_init, gamma=10e-5, eps=eps, maxiter=1000000)

print(f"Algorithme terminé en {len(iters5)} itérations")

# L'execution se passe bien pour le grand pas,
# mais la convergence est extremement lente (484 000 itération)

# %%
iters3 = grad_desc(rosen_der, x_init, gamma=10e-3, eps=eps, maxiter=30)

# Ici scipy nous alerte que l'algorithme a atteint des valeurs "absurdement" grandes, et l'algorithme diverge
print(iters3)

# %%

# Graphe de la descente de gradient et lignes de niveaux

x = np.linspace(-1.5, 1.5)
y = np.linspace(-1.5, 1.5)

X, Y = np.meshgrid(x, y)
Z = f_ros(X, Y)

N = 100  # nombre de points à afficher

plt.contourf(X, Y, Z, 50, cmap="RdBu", norm=LogNorm(),
             levels=np.logspace(-4, 4, num=50))
plt.colorbar()
plt.plot(iters5[:N, 0], iters5[:N, 1], marker='.', markersize=8,
         color="black", label="Itérés pour un pas de $10^{-5}$")
plt.scatter(-1, -0.5, color="white", edgecolor="black",
            linewidths=0.5, s=70, label="Initialisation")
plt.scatter(1, 1, label="Minimum global", marker="x", color="black")
plt.legend()

plt.title(f"{N} itérés de la descente de gradient \npour la fonction de Rosenbrock avec ""$\gamma = 10^{-5}$")

# %% [markdown]
# La descente de gradient normale ne fonctionne donc que pour certains pas. De plus la convergence de cet algo pour la fonction de Rosenbrock est extrenement lente, car une fois la vallée trouvée, (rapide : ~100 itérations ici), il faut encore ~500 000 itérations pour atteindre le minimum, à cause du faible gradient au fond de la vallée

# %% [markdown]
# # Adaptation du pas

# %%
# Question 3 : Descente de gradient avec recherche linéaire


def grad_desc_line_search(f, df, x_init, gamma, eps, maxiter):
    """Effectue une descente de gradient avec recherche linéaire.

    A chaque itération, si la fonction n'a pas assez décru (géré par 'alpha"), 
    le pas 'gamma' est multiplié ou divisé (par le facteur 'tau')    

    Renvoie la liste des x_k et des gamma_k, dans un couple de ndarrays
    """

    alpha, tau = 0.5, 0.5

    x_iters = []
    gamma_iters = []

    x = x_init

    for _ in range(maxiter):
        x_iters.append(x.copy())
        gamma_iters.append(gamma)

        g = df(x)
        norm = np.linalg.norm(g, ord=np.Inf)
        if norm <= eps:
            break
        z = x - gamma * g
        if f(z) <= f(x) - alpha * gamma * norm**2:
            x = z
            gamma /= tau
        else:
            gamma *= tau

    return np.array(x_iters), np.array(gamma_iters)


# %%
# Question 4 : Avec pas adaptatif


def grad_desc_adaptive_step(df, x_init, gamma, eps, maxiter=5000):
    """Effectue une descente de gradient avec pas adaptatif

    Renvoie la liste des x_k et des gamma_k, dans un couple de ndarrays
    """
    x = [x_init.copy()]
    gamma = [gamma, gamma]

    theta = np.Inf

    x.append(x[0] - gamma[0] * df(x[0]))

    for _ in range(2, maxiter):
        x_diff = np.linalg.norm(x[-1] - x[-2])
        grad_diff = np.linalg.norm(df(x[-1]) - df(x[-2]))
        a = sqrt(1 + theta) * gamma[-1]
        b = 0.5 * x_diff / grad_diff

        gamma.append(min(a, b))

        g = df(x[-1])
        if np.linalg.norm(g, ord=np.Inf) <= eps:
            break
        x.append(x[-1] - gamma[-1] * g)
        theta = gamma[-1]/gamma[-2]
    return np.array(x), np.array(gamma)


# %%
# Evolution de la taille des pas, exemple avec la fonction de Rosenbrock

def f_ros_np(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2) ** 2

# Execution des descentes de gradient avec les algos modifiés


N = 50  # nb d'itérations max

x_init = np.array([-1, -0.5])
gamma_initial = 10e-5
eps = 10e-10
iters_x_ls, iters_gamma_ls = grad_desc_line_search(
    f=f_ros_np, df=rosen_der, x_init=x_init, gamma=gamma_initial, eps=eps, maxiter=N)

iters_x_as, iters_gamma_as = grad_desc_adaptive_step(
    df=rosen_der, x_init=x_init, gamma=gamma_initial, eps=eps, maxiter=N)


# Graphique montrant l'évolution des gamma


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

fig.suptitle("Evolution de $\gamma_k$ pour chaque algorithme")

ax1.scatter(range(N), np.full(N, gamma_initial))
ax1.title.set_text("Descente de gradient normale")
ax1.set_ylabel(r"$\gamma_k$")

ax2.scatter(range(N), iters_gamma_ls)
ax2.set_yscale('log')
ax2.title.set_text("Descente de gradient avec recherche linéaire")
ax2.set_ylabel(r"$\gamma_k$")

ax3.scatter(range(N), iters_gamma_as)
ax3.set_yscale('log')
ax3.title.set_text("Descente de gradient avec pas adaptatif")
ax3.set_ylabel(r"$\gamma_k$")

fig.tight_layout()


# %%
# Question 5 : comparaison des convergences en fonction des itérés

# Lignes de niveaux

x = np.linspace(-1.5, 1.5)
y = np.linspace(-1.5, 1.5)

X, Y = np.meshgrid(x, y)
Z = f_ros(X, Y)

N = 30

fig = plt.figure(figsize=(8, 6))

plt.contourf(X, Y, Z, 50, cmap="RdBu", norm=LogNorm(),
             levels=np.logspace(-4, 4, num=50))
plt.colorbar()

plt.plot(iters5[:N, 0], iters5[:N, 1], marker='.', markersize=8,
         color="black", label="Itérés : descente de gradient $\gamma = 10^{-5}$")
plt.plot(iters_x_ls[:N, 0], iters_x_ls[:N, 1], marker='.', markersize=8, color="red",
         label="Itérés : descente de gradient avec recherche linéaire ($\gamma_0 = 10^{-5}$)")
plt.plot(iters_x_as[:N, 0], iters_x_as[:N, 1], marker='.', markersize=8, color="purple",
         label="Itérés : descente de gradient avec pas adaptatif ($\gamma_0 = 10^{-5}$)")

plt.scatter(-1, -0.5, color="white", edgecolor="black",
            linewidths=0.5, s=70, label="Initialisation")
plt.scatter(1, 1, label="Minimum global", marker="x", color="black")
plt.legend(loc="lower right")

plt.title(f"{N} itérés de descentes de gradient \npour la fonction de Rosenbrock")

# %% [markdown]
# On observe bien la modification de la taille des pas.
# Pour l'algo avec recherche linéaire, il faut seulement 3 étapes à l'algorithme avec recherche linéaire pour atteindre la vallée. on constate aussi que les distance entre les 4 premiers points est cohérente avec la taille des premiers gammas de l'algorithme : 0.1, 0.05, 0.025.
#
# L'algo adaptatif se rapproche encore plus du minimum, avec le même nombre d'itérations.

# %%
# Normes des gradients aux itérés

N = 50

plt.title("Norme du gradient en $x_k$")
plt.xlabel("$x_k$")
plt.ylabel("$||\\nabla f(x_k)||$")
plt.yscale(('log'))
plt.scatter(range(N), np.linalg.norm(
    rosen_der(iters5[:N, :]), axis=1), label="Descente de gradient")
plt.scatter(range(N), np.linalg.norm(
    rosen_der(iters_x_ls[:N, :]), axis=1), label="Recherche linéaire")
plt.scatter(range(N), np.linalg.norm(
    rosen_der(iters_x_as[:N, :]), axis=1), label="Adaptative")
plt.legend()


# %%
# Distances des itérés au minimum

x_min = np.array([1, 1])

plt.title("Distance à de $x_k$ à $x^\\ast$")
plt.ylabel("$||x_k-x^\\ast||$")
plt.scatter(range(N), np.linalg.norm(
    iters5[:N, :] - x_min, axis=1), label="Descente de gradient")
plt.scatter(range(N), np.linalg.norm(
    iters_x_ls[:N, :] - x_min, axis=1), label="Recherche linéaire")
plt.scatter(range(N), np.linalg.norm(
    iters_x_as[:N, :] - x_min, axis=1), label="Adaptative")
plt.yscale('log')
plt.legend()

fig.tight_layout()


# %% [markdown]
# Ce que l'on a vu sur le graphique des lignes de niveaux semble confirmé ici : les algortihmes qui se rapprochent le plus vite de la solution sont (du plus rapide au plus lent):
#     - descente adaptive
#     - recherche linéaire
#     - descente de gradient.
