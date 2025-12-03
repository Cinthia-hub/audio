import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os

# Parámetros de grabación
DURACION = 1  # segundos
FS = 16000    # frecuencia de muestreo (Hz)
ARCHIVO_SALIDA = "grabacion_prueba.wav"

print("Grabando por 2 segundos... habla ahora")
audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='int16')
sd.wait()

# Guardar archivo
wav.write(ARCHIVO_SALIDA, FS, audio)
print(f" Audio guardado como '{ARCHIVO_SALIDA}'")

# Reproducir para verificar
print(" Reproduciendo...")
sd.play(audio, FS)
sd.wait()
print(" Prueba completada.")
