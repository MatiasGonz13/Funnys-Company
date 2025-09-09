from pulp import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

I = ["Antofagasta",
    "Valparaíso",
    "Santiago",
    "Rancagua",
    "Concepción",
    "Puerto Montt"]

J = ["Pequeña", "Grande"]

K = ["R1", "R2", "R3", "R4", "R5", "R6"]

F = ["AT1", "AT2", "AT3"]

T = [1, 2, 3]

demanda_actual = {
    "R1": 951776,
    "R2": 967364,
    "R3": 512051,
    "R4": 386248,
    "R5": 946174,
    "R6": 303445
    }

tasas_crecimiento = {
    "R1": 0.16,
    "R2": 0.22,
    "R3": 0.26,
    "R4": 0.15,
    "R5": 0.39,
    "R6": 0.30
    }

capacidad_planta = {
    "Pequeña": 4636446,
    "Grande": 14966773
}

C = {
    ("Antofagasta", "Pequeña"): 86626147,
    ("Valparaíso", "Pequeña"): 115721215,
    ("Santiago", "Pequeña"): 172235977,
    ("Rancagua", "Pequeña"): 0,
    ("Concepción", "Pequeña"): 57494934,
    ("Puerto Montt", "Pequeña"): 175561277,

    ("Antofagasta", "Grande"): 201456157,
    ("Valparaíso", "Grande"): 199519337,
    ("Santiago", "Grande"): 291925385,
    ("Rancagua", "Grande"): 299031830,
    ("Concepción", "Grande"): 179671671,
    ("Puerto Montt", "Grande"): 337617842
}

Cf = {
    ("Antofagasta", "Pequeña"): 18236639,
    ("Valparaíso", "Pequeña"): 8838286,
    ("Santiago", "Pequeña"): 6840758,
    ("Rancagua", "Pequeña"): 13378246,
    ("Concepción", "Pequeña"): 26394217,
    ("Puerto Montt", "Pequeña"): 3678737,

    ("Antofagasta", "Grande"): 60788796,
    ("Valparaíso", "Grande"): 32734393,
    ("Santiago", "Grande"): 32575039,
    ("Rancagua", "Grande"): 53512984,
    ("Concepción", "Grande"): 65985543,
    ("Puerto Montt", "Grande"): 26276695
}

AT1_matrix = [
            [1.06, 2.80, 10.29, 4.87, 6.41, 10.35],
            [3.49, 6.19, 3.39, 6.77, 3.07, 6.61],
            [6.38, 5.88, 5.63, 1.01, 3.15, 5.67],
            [3.44, 1.48, 2.79, 2.80, 5.30, 1.29],
            [5.94, 7.33, 1.80, 9.48, 2.82, 8.25],
            [2.57, 9.63, 4.84, 6.64, 6.48, 8.54]
        ]

AT2_matrix = [
            [10.03, 4.09, 4.55, 7.84, 5.33, 10.63],
            [10.52, 1.82, 3.91, 8.20, 5.88, 2.33],
            [1.90, 8.89, 6.55, 9.71, 7.03, 10.23],
            [2.06, 10.17, 2.12, 6.11, 3.79, 6.19],
            [2.54, 6.95, 8.57, 10.50, 4.85, 5.31],
            [7.92, 10.32, 1.41, 4.94, 2.74, 8.08]
        ]

AT3_matrix = [
            [9.86, 4.30, 8.10, 9.63, 7.40, 6.47],
            [1.58, 2.71, 3.08, 5.91, 7.99, 5.11],
            [9.13, 10.03, 6.77, 5.70, 3.62, 8.58],
            [8.95, 7.37, 10.29, 3.34, 2.21, 4.58],
            [9.62, 3.78, 5.19, 2.61, 3.19, 1.78],
            [10.32, 8.88, 10.87, 10.38, 5.83, 1.54]
        ]

Ct = {}

for f, costos in zip(F, [AT1_matrix, AT2_matrix, AT3_matrix]):
    for i_idx, i in enumerate(I):
        for k_idx, k in enumerate(K):
            Ct[(i, k, f)] = costos[i_idx][k_idx]

Cv = {
    ("Antofagasta", "Pequeña"): 28.20, ("Antofagasta", "Grande"): 28.20,
    ("Valparaíso", "Pequeña"): 41.68, ("Valparaíso", "Grande"): 41.68,
    ("Santiago", "Pequeña"): 18.57, ("Santiago", "Grande"): 18.57,
    ("Rancagua", "Pequeña"): 17.68, ("Rancagua", "Grande"): 17.68,
    ("Concepción", "Pequeña"): 50.11, ("Concepción", "Grande"): 50.11,
    ("Puerto Montt", "Pequeña"): 43.55, ("Puerto Montt", "Grande"): 43.55
    }

D = {
    ("R1",1): 951776 * 0.16, ("R1",2): 951776 * 0.16^2, ("R1",3): 951776 * 0.16^3,
    ("R2",1): 967364 * 0.22, ("R2",2): 967364 * 0.22^2, ("R2",3): 967364 * 0.22^3,
    ("R3",1): 512051 * 0.26, ("R3",2): 512051 * 0.26^2, ("R3",3): 512051 * 0.26^3,
    ("R4",1): 386248 * 0.15, ("R4",2): 386248 * 0.15^2, ("R4",3): 386248 * 0.15^3,
    ("R5",1): 946174 * 0.39, ("R5",2): 946174 * 0.39^2, ("R5",3): 946174 * 0.39^3,
    ("R6",1): 303445 * 0.30, ("R6",2): 303445 * 0.30^2, ("R6",3): 303445 * 0.30^3,
}


#Variables de decisión
X = pulp.LpVariable.dicts(
    "X", [(i, j) for i in I for j in J], cat="Binary"
)

Y = pulp.LpVariable.dicts(
    "Y", [(i, k, f, t) for i in I for k in K for f in F for t in T], cat="Continuous"
)


#Función Objetivo
costo_fijo = pulp.lpSum([(C[(i, j)] + Cf[(i, j)]) * X[(i, j)] for i in I for j in J])

costo_variable = pulp.lpSum(
    C[(i,j)] * Y[(i,k,f,t)]
    for i in I
    for j in J
    for k in K
    for f in F
    for t in T
)

costo_transporte = pulp.lpSum(
    Ct[(i,k,f)] * Y[(i,k,f,t)]
    for i in I
    for k in K
    for f in F
    for t in T
)


prob = LpProblem("Funnys Company", LpMinimize)
prob += costo_fijo + costo_variable + costo_transporte

prob += pulp.lpSum(
    Y[(i,k,f,t)] >= D[(k,t)]
)

status = prob.solve()
LpStatus[status]
value(X)
value(Y)

# Ciudades (lon, lat)
ciudades_chile = {
    "Antofagasta": (-70.402, -23.650),
    "Santiago": (-70.648, -33.456),
    "Valparaíso": (-71.628, -33.047),
    "Concepción": (-73.049, -36.827),
    "Puerto Montt": (-72.942, -41.471),
    "Rancagua": (-70.74053, -34.1691)
}

# Figura y proyección
fig = plt.figure(figsize=(6, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Dimensiones del mapa
ax.set_extent([-75, -65, -45, -20])  

# Opciones
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgreen")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

# Mapeo
for ciudad, (lon, lat) in ciudades_chile.items():
    ax.plot(lon, lat, marker="o", color="red", markersize=6, transform=ccrs.PlateCarree())
    ax.text(lon+0.3, lat-0.3, ciudad, fontsize=8, transform=ccrs.PlateCarree())

plt.title("Ciudades bajo estudio", fontsize=12)
plt.show()


