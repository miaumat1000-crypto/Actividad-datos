# ==========================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ==========================================
# 2. CARGA Y EXPLORACIÓN INICIAL
# ==========================================
# Cargamos el dataset (asegúrate de tener el archivo CSV en la misma carpeta)
df = pd.read_csv("TikTok_songs_2020.csv")

print("Forma del dataset:", df.shape)
print("\nPrimeros 5 registros:")
print(df.head())

print("\nInformación general:")
df.info()

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nPorcentaje de nulos por columna:")
porcentaje_nulos = (df.isnull().sum() / len(df)) * 100
print(porcentaje_nulos)

print("¿Hay nulos en el dataset?", df.isnull().values.any())


# ==========================================
# 3. LIMPIEZA DE DATOS
# ==========================================
print("\nDuplicados iniciales:", df.duplicated().sum())

# Eliminar duplicados
df = df.drop_duplicates()

# Eliminar columnas técnicas que no necesitamos para este análisis
columnas_a_quitar = ["time_signature", "key"]
df = df.drop(columns=columnas_a_quitar)

print("Nuevo tamaño tras limpieza:", df.shape)
print("Columnas restantes:", df.columns.tolist())


# ==========================================
# 4. INGENIERÍA DE CARACTERÍSTICAS
# ==========================================
# Crear nueva columna: duración en minutos
df["duration_min"] = (df["duration_ms"] / 1000 / 60).round(2)

print("\nEstadísticas de duración en minutos:")
print(df["duration_min"].describe())

print("Canción más corta:", df.loc[df["duration_min"].idxmin(), "track_name"])
print("Canción más larga:", df.loc[df["duration_min"].idxmax(), "track_name"])

print("\nEstadísticas generales de columnas numéricas:")
print(df.describe().round(2))

# Calcular estadísticas específicas con NumPy
columnas_audio = ["danceability", "energy", "valence", "tempo"]
for col in columnas_audio:
    print(f"\n--- {col} ---")
    print(f"  Media: {np.mean(df[col]):.3f}")
    print(f"  Mediana: {np.median(df[col]):.3f}")
    print(f"  Desv Std: {np.std(df[col]):.3f}")
    print(f"  Min: {np.min(df[col]):.3f}")
    print(f"  Max: {np.max(df[col]):.3f}")

# Normalización Min-Max
columnas_a_normalizar = ["danceability", "energy", "valence", "tempo", "acousticness", "speechiness"]
df_normalizado = df.copy() 

for col in columnas_a_normalizar:
    minimo = df[col].min()
    maximo = df[col].max()
    df_normalizado[col + "_norm"] = (df[col] - minimo) / (maximo - minimo)

columnas_norm = [c for c in df_normalizado.columns if "_norm" in c]
print("\nPrimeras filas normalizadas:")
print(df_normalizado[columnas_norm].head())

# Cálculo Z-Score para popularidad
media_pop = np.mean(df["track_pop"])
std_pop = np.std(df["track_pop"])
df["zscore_popularidad"] = (df["track_pop"] - media_pop) / std_pop

# Detectar canciones con popularidad inusual (z > 2 o z < -2)
canciones_raras = df[np.abs(df["zscore_popularidad"]) > 2]
print("\nCanciones con popularidad inusual:")
print(canciones_raras[["track_name", "track_pop", "zscore_popularidad"]])

print("\nMás popular:", df.loc[df["track_pop"].idxmax(), "track_name"])
print("Menos popular:", df.loc[df["track_pop"].idxmin(), "track_name"])


# ==========================================
# 5. FILTRADO Y CONSULTAS AVANZADAS
# ==========================================
canciones_bailables = df[df["danceability"] > 0.8]
print(f"\nCanciones muy bailables (>0.8): {len(canciones_bailables)}")

doja_cat = df[df["artist_name"] == "Doja Cat"]
print("\nCanciones de Doja Cat:")
print(doja_cat[["track_name", "track_pop", "danceability"]])

# Top 5 más populares
df_ordenado = df.sort_values("track_pop", ascending=False)
print("\nTop 5 canciones más populares:")
print(df_ordenado[["track_name", "artist_name", "track_pop"]].head())

fiesta = df[(df["danceability"] > 0.8) & (df["energy"] > 0.7)]
print(f"\nCanciones para la fiesta (Bailables + Energéticas): {len(fiesta)}")


# ==========================================
# 6. VISUALIZACIONES (MATPLOTLIB)
# ==========================================
# Configurar estilo general
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# --- Gráfico 1: Top 10 Artistas ---
top_artistas = df.groupby("artist_name")["artist_pop"].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
colores = plt.cm.plasma(np.linspace(0.2, 0.9, 10))
ax.barh(top_artistas.index[::-1], top_artistas.values[::-1], color=colores[::-1])

ax.set_xlabel("Popularidad del Artista", fontsize=12, fontweight="bold")
ax.set_title("Top 10 Artistas más Populares en TikTok 2020", fontsize=14, fontweight="bold")

for i, val in enumerate(top_artistas.values[::-1]):
    ax.text(val + 0.5, i, f"{val:.0f}", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("top_artistas.png", dpi=150)
plt.show()

# --- Gráfico 2: Histograma de Bailabilidad ---
fig, ax = plt.subplots(figsize=(9, 5))
n, bins, patches = ax.hist(df["danceability"], bins=20, edgecolor="white", linewidth=1.2)

for i, patch in enumerate(patches):
    patch.set_facecolor(plt.cm.RdPu(i / len(patches) * 0.8 + 0.2))

promedio = df["danceability"].mean()
ax.axvline(promedio, color="#FF1493", linestyle="--", linewidth=2.5, label=f"Promedio: {promedio:.2f}")

ax.set_xlabel("Danceability (Bailabilidad)", fontsize=12)
ax.set_ylabel("Cantidad de canciones", fontsize=12)
ax.set_title("Distribución de Bailabilidad de las Canciones", fontsize=14)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("histograma_danceability.png", dpi=150)
plt.show()

# --- Gráfico 3: Scatter Plot Energía vs Valencia ---
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    df["energy"], 
    df["valence"], 
    c=df["track_pop"], 
    cmap="YlOrRd", 
    s=60, 
    alpha=0.75, 
    edgecolors="white", 
    linewidth=0.5
)

barra_color = plt.colorbar(scatter, ax=ax)
barra_color.set_label("Popularidad de la canción", fontsize=11)

ax.set_xlabel("Energía", fontsize=12, fontweight="bold")
ax.set_ylabel("Valencia (qué tan feliz suena)", fontsize=12)
ax.set_title("Energía vs Felicidad de las Canciones de TikTok", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("scatter_energia_valencia.png", dpi=150)
plt.show()

# --- Gráfico 4: Gráfico de Pastel (Modo Mayor vs Menor) ---
fig, ax = plt.subplots(figsize=(7, 7))
conteo_modo = df["mode"].value_counts()
etiquetas = ["Mayor (Alegre)", "Menor (Melancolico)"]
colores_pastel = ["#FFD700", "#9B59B6"]  # Amarillo y morado

wedges, texts, autotexts = ax.pie(
    conteo_modo.values,
    labels=etiquetas,
    colors=colores_pastel,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)

for autotext in autotexts:
    autotext.set_fontweight("bold")
    autotext.set_fontsize(13)

ax.set_title("Canciones en Modo Mayor vs Menor", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("pie_modo.png", dpi=150)
plt.show()

