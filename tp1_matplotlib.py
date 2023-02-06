import matplotlib.pyplot as plt
import numpy as np

# %%
n = 1000
a, b = -10, 10
x = np.linspace(a, b, n)

plt.figure(figsize=(10, 5))
plt.plot(x, np.cos(x), zorder=1, label=r"$f:x\mapsto\cos(x)$")
plt.hlines(y=(-1, 1),
           xmin=a,
           xmax=b,
           color="red",
           linestyle="dotted",
           zorder=2,
           label=f"$\\pm {1}$",
           linewidth=3
           )

x_extrema = np.arange(-3, 4) * np.pi
y_extrema = np.cos(x_extrema)

plt.scatter(x_extrema, y_extrema, s=24, label="extremas")

plt.legend()
plt.title(f"Fonction sinus sur [{a}, {b}]")
plt.ylabel("sin(x)")
plt.xlabel("x")
plt.show()
# %% Graphes multiples

fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

colors = plt.cm.Purples(np.linspace(0.4, 1, 5))

n = 1000
x = np.linspace(0, 10, n, endpoint=True)

for i in range(1, 6):
    y = np.exp(-i * x)
    ax1.plot(x, y, label=f"$\\lambda = {i} $", color=colors[i-1])

    y = np.exp(-i * x)
    ax2.semilogy(x, y, label=f"$\\lambda = {i} $", color=colors[i-1])

ax1.legend()
ax1.set_title("Echelle linéaire")

ax2.legend()
ax2.set_title("Echelle semi-logarithmique")
fig1.suptitle("Décroissance exponentielle")
fig1.show()
# %%
# Plusieurs variables
x = np.linspace(-10, 10, 11)
y = np.linspace(0, 20, 11)
xx, yy = np.meshgrid(x, y)

fig_level_set, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(xx, yy, ls="None", marker='.')

ax1.set_aspect('equal')  # même échelle sur les deux axes
ax2.plot(yy, xx, ls="None", marker='.')
ax2.set_aspect('equal')  # même échelle sur les deux axes

fig_level_set.show()


# # %%
# # conda install ipympl
# from IPython import get_ipython  # noqa
# get_ipython().run_line_magic("matplotlib", "widget")

# %%


def f(x, y):
    return x**2 - y**4

# %%
# Lignes de niveaux simples


x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.suptitle("Deux méthodes de lignes de niveau de $x^2 - y^4$ sur $[-1,1]^2$")

plt.subplot(121)

plt.contour(X, Y, Z, 40, cmap='RdGy', figsize=(5, 5))  # red gray, 40 lines
plt.title("plt.contour")
plt.colorbar()

plt.subplot(122)

plt.contourf(X, Y, Z, 40, cmap='RdGy', figsize=(5, 5))
plt.title("plt.contourf")
plt.colorbar()
plt.tight_layout()
plt.show()
# %%
# retour au tp :


x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# lignes de niveau
fig_level_sets, ax_level_sets = plt.subplots(1, 1, figsize=(4, 4))
ax_level_sets.set_title("Lignes de niveau de $x^2 - y^4$")
level_sets = ax_level_sets.contourf(
    X, Y, Z, levels=30, cmap="RdBu_r")  # red blue reversed
fig_level_sets.colorbar(level_sets,
                        ax=ax_level_sets,
                        fraction=0.04,  # fraction of original axes to use for colorbar
                        # (padding) fraction of original axes between colorbar and image
                        pad=0.1
                        )

ax_level_sets.scatter([0], [0], color="grey", s=5)

# surface

fig_surface, ax_surface = plt.subplots(1, 1,
                                       subplot_kw={"projection": "3d"}
                                       )
ax_surface.set_title("Surface de $x^2 - y^4$")
surf = ax_surface.plot_surface(X, Y, Z,
                               rstride=1,  # row step size
                               cstride=1,  # column step size
                               cmap=plt.cm.RdBu_r,
                               #    linewidth=10, ???
                               alpha=0.8,
                               antialiased=False

                               )
ax_surface.scatter3D([0], [0], [0], color="black", s=20)

plt.show()

# On remarque que s'il l'on se déplace le long de l'axe des abscisses,
# on se "rapproche du rouge" et donc la fonction partielle en cette direction
# est croissante. A l'inverse, si l'on se déplace le long de l'axe
# des ordonnées, on se "rapproche du bleu" et donc la fonction
# partielle en cette direction est décroissante.
# L'origine n'est donc pas un minimum local ni un maximum local.
# Pourtant, f''(0,0) = [2, 0] est semi définie positive et f'(0,0) = 0
#                      [0, 0]
#

# %% Dérivées partielles


f_tex = "$\\frac{xy}{x^2 + y^2}$"


def f(x, y):
    return np.where(np.allclose([x, y], 0), 0, x*y/(x**2 + y**2))


def dx(x, y):
    return np.where(np.allclose([x, y], 0), 0, y*(y**2 - x**2)/(x**2+y**2)**2)


def dy(x, y):
    return np.where(np.allclose([x, y], 0), 0, x*(x**2 - y**2)/(x**2+y**2)**2)


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

dX = dx(X, Y)
dY = dy(X, Y)
speed = np.sqrt(dX*dX + dY*dY)


fig1 = plt.figure(figsize=(8, 4))

# Lignes de niveau
ax1 = fig1.add_subplot(1, 2, 1)
im = ax1.contourf(X, Y, Z, 30, cmap='RdBu_r', figsize=(10, 10))
ax1.streamplot(X, Y, dX, dY, ccolor='k', linewidth=5*speed/speed.max())

# L'épaisseur des flèches nous indique que les pentes sont plus fortes
# au centre de la surface

ax1.set_xlim([x.min(), x.max()])
ax1.set_ylim([y.min(), y.max()])
ax1.set_aspect('equal')
cbar = fig1.colorbar(im, ax=ax1, fraction=0.045, pad=0.04)
ax1.set_title(f'Lignes de niveau de  {f_tex}')

# Surface
ax2 = fig1.add_subplot(1, 2, 2, projection="3d")
ax2.set_title(f"Surface de {f_tex}")
ax2.plot_surface(X, Y, Z,
                 cmap=plt.cm.RdBu_r,
                 cstride=1,
                 rstride=1,
                 linewidth=0,
                 antialiased=False,
                 )

plt.tight_layout()
plt.show()
# %% Variante
fig2, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.imshow(Z,
          origin="lower",  # car sinon l'axe des ordonnées est inversé
          extent=[min(x), max(x), min(y), max(y)],
          cmap=plt.cm.RdBu_r,
          )


CS = ax.contour(X, Y, Z, colors="white", linewidths=0.8)
ax.clabel(CS,  # contour label
          fontsize=10)
plt.tight_layout(pad=3)
plt.show()

# On remarque que les lignes de niveau passent par l'origine.
# Cela veut dire que les fonctions partielles selon les droites
# passant par l'origine sont constantes, et donc le gradient est nul
# en (0,0)
