# -----------------------------------------------------------------------------------
# Función: extraer_features
# Esta función extrae un conjunto enriquecido de características acústicas desde un 
# archivo de audio (.wav) para usar en tareas de clasificación de comandos de voz.
#
# Características extraídas:
# 
# 1. MFCCs (Mel-Frequency Cepstral Coefficients) - 13 coeficientes:
#    Capturan la envolvente del espectro en escala mel, modelando cómo el oído humano
#    percibe el sonido. Se calculan la media y desviación estándar de cada coeficiente,
#    lo que permite representar tanto la forma general del espectro como su variabilidad.
#    Total: 13 (media) + 13 (std) = 26 características.
#
# 2. RMS Energy (Root Mean Square) - 1 valor (media) + 1 valor (std):
#    Representa la energía del sonido. Ayuda a distinguir entre sonidos fuertes y débiles
#    o silencios. La combinación de media y std capta tanto el nivel general como la 
#    dinámica de energía.
#
# 3. Spectral Centroid - 1 valor (media):
#    Indica la "brillantez" del sonido (centro de masa espectral). Útil para diferenciar
#    sonidos agudos de graves.
#
# 4. Spectral Bandwidth - 1 valor (media):
#    Mide la amplitud del espectro. Útil para distinguir sonidos "anchos" (ruido, fricativas)
#    de sonidos más puros.
#
# 5. Zero Crossing Rate - 1 valor (media):
#    Frecuencia con la que la señal cruza el eje cero. Es útil para medir la aspereza o
#    complejidad de un sonido.
#
# 6. Chroma STFT - 12 valores (media):
#    Representa la energía distribuida en 12 clases de altura tonal (Do, Do#, Re, ... Si).
#    Útil para capturar información tonal y de armonía, especialmente en señales vocales.
#
# El vector resultante tiene un total de 43 características por muestra de audio.
# Estas fueron elegidas por su capacidad de representar tanto el contenido espectral
# como dinámico y tonal del habla, manteniendo un tamaño reducido adecuado para redes simples.
# -----------------------------------------------------------------------------------


import os
import librosa
import numpy as np
import csv

# Parámetros
CARPETA_COMANDOS = "pruebas_audio"
N_MUESTRAS_POR_CLASE = 900
N_MFCC = 13
ARCHIVO_SALIDA = "features_test.csv"

def extraer_features(audio_path, sr=16000, n_mfcc=N_MFCC):
    y, sr = librosa.load(audio_path, sr=sr)
    y, _ = librosa.effects.trim(y)

    # MFCC y estadísticas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral centroid y bandwidth
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Combinar todo
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [rms_mean, rms_std],
        np.mean(centroid, axis=1),
        np.mean(bandwidth, axis=1),
        np.mean(zcr, axis=1),
        chroma_mean
    ])

    return features  # vector plano

def cargar_datos(carpeta):
    data = []
    etiquetas = sorted(os.listdir(carpeta))  # Mantiene orden alfabético

    for etiqueta in etiquetas:
        carpeta_clase = os.path.join(carpeta, etiqueta)
        if not os.path.isdir(carpeta_clase):
            continue

        archivos = [f for f in os.listdir(carpeta_clase) if f.endswith(".wav")]
        archivos = archivos[:N_MUESTRAS_POR_CLASE]

        print(f"Procesando {len(archivos)} archivos de '{etiqueta}'...")

        for archivo in archivos:
            path_audio = os.path.join(carpeta_clase, archivo)
            try:
                features = extraer_features(path_audio)
                fila = list(features) + [etiqueta]
                data.append(fila)
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")

    return data

if __name__ == "__main__":
    datos = cargar_datos(CARPETA_COMANDOS)

    # Generar encabezados dinámicos
    num_features = len(datos[0]) - 1  # excluye label
    encabezado = [f"feat_{i}" for i in range(num_features)] + ["label"]

    # Guardar CSV
    with open(ARCHIVO_SALIDA, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(encabezado)
        writer.writerows(datos)

    print(f"¡Extracción completada y guardada en '{ARCHIVO_SALIDA}'!")