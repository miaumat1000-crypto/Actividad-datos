import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar el dataset de canciones de TikTok
df = pd.read_csv("TikTok_songs_2020.csv")

# Configuración visual para los gráficos de Matplotlib
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("=" * 60)
print("              RESOLUCIÓN DE LOS 4 RETOS FINALES")
print("=" * 60)

# ==============================================================================
# RETO 1: Top 5 canciones con mayor "speechiness" + Gráfica horizontal
# ==============================================================================
top5_speechiness = df.sort_values(by="speechiness", ascending=False)[
    ["track_name", "artist_name", "speechiness"]
].head(5)

print("\n--- RETO 1: TOP 5 CANCIONES CON MAYOR SPEECHINESS ---")
print(top5_speechiness.to_string(index=False))

# Gráfica Reto 1
fig, ax = plt.subplots(figsize=(9, 4.5))
colores_r1 = plt.cm.spring(np.linspace(0.2, 0.8, 5))
etiquetas_r1 = [f"{row['track_name']}\n({row['artist_name']})" for _, row in top5_speechiness.iterrows()]

ax.barh(etiquetas_r1[::-1], top5_speechiness["speechiness"][::-1], color=colores_r1)
ax.set_xlabel("Nivel de Speechiness (Voz Hablada)", fontsize=11, fontweight="bold")
ax.set_ylabel("Canción (Artista)", fontsize=11, fontweight="bold")
ax.set_title("Reto 1: Top 5 Canciones con Mayor 'Speechiness'", fontsize=13, fontweight="bold")

for i, val in enumerate(top5_speechiness["speechiness"][::-1]):
    ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold")

plt.tight_layout()
plt.show()

# ==============================================================================
# RETO 2: Clasificación de canciones por categoría de energía
# ==============================================================================
df["categoria_energia"] = np.where(
    df["energy"] < 0.4, "Baja", 
    np.where(df["energy"] <= 0.7, "Media", "Alta")
)

conteo_energia = df["categoria_energia"].value_counts()

print("\n--- RETO 2: DISTRIBUCIÓN POR CATEGORÍA DE ENERGÍA ---")
print(conteo_energia.to_string())

# ==============================================================================
# RETO 3: Top 10 artistas con más canciones en el dataset + Gráfica
# ==============================================================================
top10_artistas = df["artist_name"].value_counts().head(10)

print("\n--- RETO 3: TOP 10 ARTISTAS CON MÁS CANCIONES ---")
print(top10_artistas.to_string())

# Gráfica Reto 3
fig, ax = plt.subplots(figsize=(9, 4.5))
colores_r3 = plt.cm.plasma(np.linspace(0.2, 0.9, 10))

ax.barh(top10_artistas.index[::-1], top10_artistas.values[::-1], color=colores_r3)
ax.set_xlabel("Cantidad de Canciones en el Dataset", fontsize=11, fontweight="bold")
ax.set_ylabel("Artista", fontsize=11, fontweight="bold")
ax.set_title("Reto 3: Top 10 Artistas con Más Canciones", fontsize=13, fontweight="bold")

for i, val in enumerate(top10_artistas.values[::-1]):
    ax.text(val + 0.1, i, str(val), va="center", fontweight="bold")

plt.tight_layout()
plt.show()

# ==============================================================================
# RETO 4: Correlación entre Danceability y Valence
# ==============================================================================
correlacion = np.corrcoef(df["danceability"], df["valence"])[0, 1]

print("\n--- RETO 4: CORRELACIÓN DANCEABILITY VS VALENCE ---")
print(f"Coeficiente de correlación de Pearson (r): {correlacion:.4f}")
print("=" * 60)