import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pulp import *
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cartopy.feature as cfeature

img_petit =mpimg.imread('Taller de Progra/statics/fabrica_pequeña.png')
img_grande =mpimg.imread('Taller de Progra/statics/fabrica_grande.png')

def plot_factory_pequeña(ax, lon, lat, zoom=0.05):
    """Dibuja una fábrica en (lon, lat)"""
    imagebox = OffsetImage(img_petit, zoom=zoom)
    ab = AnnotationBbox(imagebox, (lon, lat), frameon=False, 
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))  
    ax.add_artist(ab)

def plot_factory_grande(ax, lon, lat, zoom=0.05):
    """Dibuja una fábrica en (lon, lat)"""
    imagebox = OffsetImage(img_grande, zoom=zoom)
    ab = AnnotationBbox(imagebox, (lon, lat), frameon=False, 
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))  
    ax.add_artist(ab)

# Conjuntos
I = ["Antofagasta", "Valparaíso", "Santiago", "Rancagua", "Concepción", "Puerto Montt"]
J = ["Pequeña", "Grande"]
K = ["R1", "R2", "R3", "R4", "R5", "R6"]
F = ["AT1", "AT2", "AT3"]
T = [1, 2, 3]

# Demanda y crecimiento
demanda_actual = {"R1": 951776, "R2": 967364, "R3": 512051, "R4": 386248, "R5": 946174, "R6": 303445}
tasas_crecimiento = {"R1": 0.16, "R2": 0.22, "R3": 0.26, "R4": 0.15, "R5": 0.39, "R6": 0.30}

# Capacidades
P = {("Pequeña", t): 4636446 for t in T}
P.update({("Grande", t): 14966773 for t in T})

# Costos fijos
C = {
    ("Antofagasta", "Pequeña"): 86626147, ("Valparaíso", "Pequeña"): 115721215,
    ("Santiago", "Pequeña"): 172235977, ("Rancagua", "Pequeña"): 0,
    ("Concepción", "Pequeña"): 57494934, ("Puerto Montt", "Pequeña"): 175561277,
    ("Antofagasta", "Grande"): 201456157, ("Valparaíso", "Grande"): 199519337,
    ("Santiago", "Grande"): 291925385, ("Rancagua", "Grande"): 299031830,
    ("Concepción", "Grande"): 179671671, ("Puerto Montt", "Grande"): 337617842
}

# Costos fijos adicionales
Cf = {
    ("Antofagasta", "Pequeña"): 18236639, ("Valparaíso", "Pequeña"): 8838286,
    ("Santiago", "Pequeña"): 6840758, ("Rancagua", "Pequeña"): 13378246,
    ("Concepción", "Pequeña"): 26394217, ("Puerto Montt", "Pequeña"): 3678737,
    ("Antofagasta", "Grande"): 60788796, ("Valparaíso", "Grande"): 32734393,
    ("Santiago", "Grande"): 32575039, ("Rancagua", "Grande"): 53512984,
    ("Concepción", "Grande"): 65985543, ("Puerto Montt", "Grande"): 26276695
}

# Costos variables por planta
Cv = {
    ("Antofagasta", "Pequeña"): 28.20, ("Antofagasta", "Grande"): 28.20,
    ("Valparaíso", "Pequeña"): 41.68, ("Valparaíso", "Grande"): 41.68,
    ("Santiago", "Pequeña"): 18.57, ("Santiago", "Grande"): 18.57,
    ("Rancagua", "Pequeña"): 17.68, ("Rancagua", "Grande"): 17.68,
    ("Concepción", "Pequeña"): 50.11, ("Concepción", "Grande"): 50.11,
    ("Puerto Montt", "Pequeña"): 43.55, ("Puerto Montt", "Grande"): 43.55
}

# Costos de transporte
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
            Ct[(i,k,f)] = int(costos[i_idx][k_idx])

# Demanda proyectada (crecimiento compuesto)
D = {}
for k in K:
    for t in T:
        D[(k,t)] = demanda_actual[k] * ((1 + tasas_crecimiento[k]) ** t)

# Variables de decisión

# Cantidad de unidades transportadas desde la ciudad i a la region k en el tipo de transporte f en el año t
Y = LpVariable.dicts("Unidades Transportadas", [(i,k,f,t) for i in I for k in K for f in F for t in T], lowBound=0, cat="Integer")

# Si se abre una planta del tipo j en la ciudad i
X = LpVariable.dicts("Apertura Planta", [(i,j) for i in I for j in J], cat="Binary")

# Modelo y función objetivo
prob = LpProblem("Funnys_Company", LpMinimize)

costo_fijo = lpSum(
    (C[(i,j)] + Cf[(i,j)]) * X[(i,j)]
    for i in I
    for j in J
    )

costo_variable = lpSum(
    lpSum(
        Cv[(i,j)]
        for j in J
        ) *
    lpSum(
        Y[(i,k,f,t)]
        for k in K
        for f in F
        for t in T)
    for i in I
    )

costo_transporte = lpSum(
    Ct[(i,k,f)] * lpSum(
        Y[(i,k,f,t)]
        for t in T
        )
    for i in I
    for k in K
    for f in F
    )

prob += costo_fijo + costo_variable + costo_transporte

## Restricciones del problema

# Lo que se produce, se envía
for t in T:
  for i in I:
    prob += lpSum(P[(j,t)] * X[(i,j)] for j in J) >= lpSum(Y[(i,k,f,t)] for k in K for f in F)

# Satisfaciión de demandas
for k in K:
    for t in T:
        prob += lpSum(Y[(i,k,f,t)] for i in I for f in F) >= D[(k,t)]

# Capacidad de producción
for i in I:
  prob += lpSum(Y[(i,k,f,t)] for j in J for k in K for f in F) <= lpSum(P[(j,t)] * X[(i,j)] for t in T for j in J)


## Restricciones de supuestos

# Capacidad de plantas en una sola ciudad
for i in I:
    prob += X[(i,"Pequeña")] + X[(i,"Grande")] <= 1


## Análisis de sensibilidad

#prob += lpSum(X[("Antofagasta", j)] for j in J) == 0
#prob += lpSum(X[("Valparaíso", j)] for j in J) == 0
#prob += lpSum(X[("Santiago", j)] for j in J) == 0
#prob += lpSum(X[("Rancagua", j)] for j in J) == 0
#prob += lpSum(X[("Concepción", j)] for j in J) == 0
#prob += lpSum(X[("Puerto Montt", j)] for j in J) == 0

# Solver
status = prob.solve(PULP_CBC_CMD(mip=True))
print("Status:", LpStatus[status])

counter = 0
total = ""
for digit in str(round(value(prob.objective)))[::-1]:
  if counter == 3:
    total+="."
    total+=digit
    counter = 1
  else:
    total+=digit
    counter+=1
print("Costo total:", total[::-1], "\n")

print("Diferencia entre Costo total redondeado y el Costo total Real:", abs(round(value(prob.objective)) - value(prob.objective)), "\n")

# Ciudades (lon, lat)
ciudades_chile = {
    "Antofagasta": (-70.402, -23.650),
    "Santiago": (-70.648, -33.456),
    "Valparaíso": (-71.628, -33.047),
    "Concepción": (-73.049, -36.827),
    "Puerto Montt": (-72.942, -41.471),
    "Rancagua": (-70.74053, -34.1691)
}

ciudades_elegidas = []
plantas_elegidas = []
for v in prob.variables():
  if int(v.varValue) != 0.0:
    if v.name.startswith("Apertura_Planta"):
      ciudades_elegidas.append(v.name.split("_")[2][2:-2])
      plantas_elegidas.append(v.name.split("_")[3][1:-2])
    print(v.name, ":" , v.varValue)

# Ciudades (lon, lat)
ciudades_chile = {
    "Antofagasta": (-70.402, -23.650),
    "Santiago": (-70.648, -33.456),
    "Valparaíso": (-71.628, -33.047),
    "Concepción": (-73.049, -36.827),
    "Puerto Montt": (-72.942, -41.471),
    "Rancagua": (-70.74053, -34.1691)
}

## Mapa General
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
    if ciudad in ciudades_elegidas:
        ax.plot(lon, lat, marker="o", color="red", markersize=10, transform=ccrs.PlateCarree())
        ax.text(lon+0.3, lat-0.3, ciudad, fontsize=8, transform=ccrs.PlateCarree())
    else:
      ax.plot(lon, lat, marker="x", color="red", markersize=6, transform=ccrs.PlateCarree())
      ax.text(lon+0.3, lat-0.3, ciudad, fontsize=8, transform=ccrs.PlateCarree())

plt.title("Ciudades bajo estudio", fontsize=12)

## Mapa Detallado
# -----------------------------
# 1️⃣ Leer shapefile
# -----------------------------
# Use the correct path from the available files
gdf_santiago = gpd.read_file("Taller de Progra\Areas_Pobladas\Areas_Pobladas.shp")

# -----------------------------
# 2️⃣ Reproyectar a lat/lon (EPSG:4326)
# -----------------------------
gdf_santiago = gdf_santiago.to_crs("EPSG:4326")

# Crear mapa general
fig, ax = plt.subplots(figsize=(8, 8),
                       subplot_kw={"projection": ccrs.PlateCarree()})

# Fondo
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="lightgrey")
ax.add_feature(cfeature.RIVERS, linewidth=0.5)
ax.add_feature(cfeature.LAKES, linewidth=0.5)

# Guardar coordenadas de las ciudades para calcular el zoom
all_lons, all_lats = [], []

# Dibujar todas las ciudades elegidas
i = 0
for ciudad in ciudades_elegidas:
    row = gdf_santiago[gdf_santiago["Localidad"] == ciudad].dissolve(by="Localidad")
    if row.empty:
        continue
    
    geom = row.geometry.iloc[0]
    lon, lat = geom.centroid.x, geom.centroid.y

    # Polígono de la ciudad
    ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                      facecolor="lightblue", edgecolor="black", linewidth=1)

    # Fábrica
    if plantas_elegidas[i] == "Pequeña":
        plot_factory_pequeña(ax, lon, lat, zoom=0.06)
    if plantas_elegidas[i] == "Grande":
        plot_factory_grande(ax, lon, lat, zoom=0.06)

    # Nombre
    ax.text(lon, lat-0.05, ciudad, ha="center", fontsize=9,
            transform=ccrs.PlateCarree(), weight="bold")

    all_lons.append(lon)
    all_lats.append(lat)

    i +=1

# Ajustar el zoom automático a todas las ciudades
if all_lons and all_lats:
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    ax.set_extent([min_lon-0.3, max_lon+0.3, min_lat-0.3, max_lat+0.3])

plt.title("Mapa de todas las fábricas construidas")

plt.show()


