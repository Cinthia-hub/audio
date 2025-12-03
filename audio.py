import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo de audio
audio_path = "comandos/four/0b40aa8e_nohash_0.wav"  # Reemplaza con la ruta de tu archivo de audio
y, sr = librosa.load(audio_path, sr=None)  # y: señal de audio, sr: sample rate
y, _ = librosa.effects.trim(y, top_db=25)  # Elimina silencios por debajo de -20 dB

# Calcular MFCC
mfcc = librosa.feature.mfcc(
    y=y,
    sr=16000,
    n_mfcc=15,
    n_fft=400,
    hop_length=160,
    n_mels=40,
    fmin=20,
    fmax=8000
)
# Calcular energía RMS
rms = librosa.feature.rms(
    y=y,               # La señal de audio
    frame_length=2048, # Tamaño de la ventana (puedes ajustarlo)
    hop_length=512     # Cuánto avanzar entre ventanas
)


# Mostrar duración
duracion = librosa.get_duration(y=y, sr=sr)
print(f"Duración del audio: {duracion:.2f} segundos")
print(f"Forma de los MFCC: {mfcc.shape}")
print(f"Forma de la energía RMS: {rms.shape}")

print(f"MFCC: {mfcc}")
print(f"Energía RMS: {rms}")

# Graficar señal, MFCC y energía
plt.figure(figsize=(12, 8))

# Señal de audio
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Señal de Audio')

# MFCC
plt.subplot(3, 1, 2)
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')

# Energía RMS
plt.subplot(3, 1, 3)
frames = range(len(rms[0]))
t = librosa.frames_to_time(frames, sr=sr)
plt.plot(t, rms[0])
plt.title('Energía RMS')
plt.xlabel('Tiempo (s)')

plt.tight_layout()
plt.show()
