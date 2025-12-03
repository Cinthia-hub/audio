import librosa
import soundfile as sf
import os
import numpy as np
import random

def segmentar_ruido_aug(archivo_ruido, carpeta_salida, duracion_segmento=1.0):
    os.makedirs(carpeta_salida, exist_ok=True)
    
    y, sr = librosa.load(archivo_ruido, sr=None)
    segmento_muestras = int(duracion_segmento * sr)

    total_segmentos = (len(y) - segmento_muestras) // segmento_muestras

    for i in range(total_segmentos):
        # Desplazamiento aleatorio (hasta 200 ms en ambas direcciones)
        jitter = random.randint(-int(0.2 * sr), int(0.2 * sr))
        inicio = max(0, i * segmento_muestras + jitter)
        fin = inicio + segmento_muestras

        if fin > len(y):  # Evita cortar fuera del audio
            break

        segmento = y[inicio:fin]

        # Aumento: cambio de ganancia ±10%
        gain = random.uniform(0.9, 1.1)
        segmento *= gain

        # Aumento: añadir ruido blanco leve
        noise = np.random.normal(0, 0.001, size=segmento.shape)
        segmento += noise

        # Normalizar para evitar clipping
        max_val = np.max(np.abs(segmento))
        if max_val > 1:
            segmento = segmento / max_val

        # Guardar archivo
        nombre_archivo = f"silent_aug_r{i:03}.wav"
        ruta_archivo = os.path.join(carpeta_salida, nombre_archivo)
        sf.write(ruta_archivo, segmento, sr)

    print(f"{i+1} segmentos aumentados de {duracion_segmento} segundos guardados en '{carpeta_salida}'.")

archivo_ruido = "comandos/back/white_noise.wav"
carpeta_salida = "comandos/silent/"

segmentar_ruido_aug(archivo_ruido, carpeta_salida)
