# Solucion del ejercicio:

## Reto 1:

El ejercicio indica que tenemos que encontrar las 5 canciones con mayor speechiness (mas habladas que cantadas, con su respectivo artista).

Para ejecutar la actividad realizaremos una liea de codigo capaz de identificar los artistas con los valores mas altos en la columna de speechiness. Con los datos establecidos se podra generar una grafica de manera decendente en donde se eviencien estos cambios de manera mas clara.

Con la linea: 
    
    top5_speechiness = df.sort_values(by="speechiness", ascending=False)[["track_name", "artist_name", "speechiness"]].head(5)

    Se exporta un listado de los artistas con mayor speechiness de menor a mayor.

Despues con el bloque:

        fig, ax = plt.subplots(figsize=(10, 5))
        colores_r1 = plt.cm.spring(np.linspace(0.2, 0.8, 5))

        etiquetas_r1 = [f"{row['track_name']}\n({row['artist_name']})" for _, row in top5_speechiness.iterrows()]

        ax.barh(etiquetas_r1[::-1], top5_speechiness["speechiness"][::-1], color=colores_r1)
        ax.set_xlabel("Nivel de Speechiness (Voz Hablada)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Canción (Artista)", fontsize=12, fontweight="bold")
        ax.set_title("Top 5 Canciones con Mayor 'Speechiness' en TikTok 2020", fontsize=14, fontweight="bold")

        for i, val in enumerate(top5_speechiness["speechiness"][::-1]):
            ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig("reto1_speechiness_grafica.png", dpi=150)
        plt.show()
    
* Se establecen la distribucion del diagrama
* Se establecen los titulos 
* Grafica las barras de manera horizontal de manera que las barras queden mirando hacia arriba
* Agrega el nivel numerico de cada barra

Conclusión: Las canciones virales con mayor speechiness presentan un formato donde predomina la voz hablada o fragmentos narrados/estrofas habladas sobre la melodía cantada, lo que refleja la gran popularidad de audios cómicos o de tendencias habladas dentro de TikTok.

## Reto 2:

Para generar una nueva columna, utilizamos las funciones condicionales np.where() para evaluar el valor de la columna energy: si es menor a 0.4 se asigna "Baja", si está entre 0.4 y 0.7 se asigna "Media", y si es mayor a 0.7 se asigna "Al

    import numpy as np

    # Asignamos la categoría según los rangos de energía
    df["categoria_energia"] = np.where(df["energy"] < 0.4, "Baja", 
                                    np.where(df["energy"] <= 0.7, "Media", "Alta"))

    # Verificamos la distribución de canciones por categoría
    print(df["categoria_energia"].value_counts())

Conclusión: La gran mayoría de las canciones virales de TikTok en 2020 tienen un nivel de energía medio a alto (268 de 292), lo que evidencia que la plataforma privilegia ritmos dinámicos y activos sobre pistas suaves o lentas.
## Reto 3: 

Para este reto se necesitan encontrar a los 10 artistas que aparezcan con mas frecuencia; posteriormente, graficar los valores obtenidos.

Usamos .value_counts() en la columna artist_name para contar la frecuencia con la que aparece cada artista en el dataset y seleccionamos los primeros 10 con .head(10). Luego, construimos una gráfica de barras horizontales con matplotlib.

    top10_frecuencia = df["artist_name"].value_counts().head(10)

    
    plt.figure(figsize=(10, 5))
    colores = plt.cm.plasma(np.linspace(0.2, 0.9, 10))
    plt.barh(top10_frecuencia.index[::-1], top10_frecuencia.values[::-1], color=colores)

    plt.xlabel("Cantidad de Canciones", fontsize=12, fontweight="bold")
    plt.ylabel("Artista", fontsize=12, fontweight="bold")
    plt.title("Top 10 Artistas con Más Canciones en TikTok 2020", fontsize=14, fontweight="bold")

    for i, val in enumerate(top10_frecuencia.values[::-1]):
        plt.text(val + 0.1, i, str(val), va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("reto3_artistas_frecuencia.png", dpi=150)
    plt.show()

* Contamos la presencia de cada artista
* Generamos el gráfico de barras horizontales

Conclusión: Doja Cat domina la plataforma en volumen de contenidos con 10 canciones en la lista, duplicando al segundo lugar (Lady Gaga con 5), demostrando que fue la artista con mayor impacto recurrente en los trends de 2020.

## Reto 4:

Este punto tiene una instruccion bastane especifica. Pide relacionar si las canciones mas bailables tienen relacion con las mas felices.

Para responder esto sin adivinar, usamos un concepto estadístico llamado coeficiente de correlación de Pearson.  

¿Qué significan las variables?Danceability (Bailabilidad): 

Mide qué tan apta es una canción para bailar (ritmo, estabilidad, fuerza del pulso) en una escala de 0 a 1.

Valence (Valencia): Mide la "positividad musical" que transmite el tema. Valores cercanos a 1 significan que la canción suena alegre, eufórica o feliz; valores cercanos a 0 indican que suena triste, melancólica o enojada. 

¿Cómo funciona np.corrcoef()?

La función de NumPy calcula una matriz de correlación que te devuelve un número entre -1 y 1:  
Cercano a +1: 
Correlación positiva fuerte (si sube la bailabilidad, la felicidad también sube mucho).
Cercano a 0: Sin correlación o correlación muy débil (las dos variables no tienen una relación clara).
Cercano a -1: Correlación negativa (si sube la bailabilidad, la felicidad baja).

Entonces, usamos la función np.corrcoef() pasando las dos columnas de interés para calcular el coeficiente de correlación de Pearson.

    matriz_correlacion = np.corrcoef(df["danceability"], df["valence"])
    r = matriz_correlacion[0, 1]
    print(f"El coeficiente de correlación es: {r:.4f}")

Conclusión: La correlación es débilmente positiva ($r \approx 0.27$), lo que indica que aunque una canción sea muy bailable, no necesariamente tiene que sonar feliz o alegre; existen muchas canciones bailables con tonos melancólicos o de baja valencia.
