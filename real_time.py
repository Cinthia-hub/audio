import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa
import tensorflow as tf
import joblib
import cv2

FS = 16000
DURACION = 2  # segundos
N_MFCC = 13
MODELO_PATH = "modelo_audio.keras"
SCALER_PATH = "scaler.pkl"
ETIQUETAS = ['one', 'two', 'three', 'four', 'five']

def extraer_features_desde_array(y, sr=FS):
    y, _ = librosa.effects.trim(y)
    
    if len(y) == 0 or np.max(np.abs(y)) < 1e-5:
        raise ValueError("Señal vacía o demasiado silenciosa")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        [np.mean(rms), np.std(rms)],
        [np.mean(centroid), np.mean(bandwidth)],
        [np.mean(zcr)],
        np.mean(chroma, axis=1)
    ])
    return features

# Cargar modelo y scaler
modelo = tf.keras.models.load_model(MODELO_PATH)
scaler = joblib.load(SCALER_PATH)

print("Presiona la tecla 'g' para grabar 2 segundos de audio...")
#crear ventana de OpenCV
cv2.namedWindow("Grabacion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Grabacion", 400, 120)
#mostrar la ventana

while True:
    
    cv2.imshow("Grabacion", 255 * np.ones((120, 400, 3), dtype=np.uint8))
    cv2.putText(255 * np.ones((120, 400, 3), dtype=np.uint8), "Presiona 'g' para grabar", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    cv2.waitKey(1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        # Mostrar ventana de grabación
        ventana = 255 * np.ones((120, 400, 3), dtype=np.uint8)
        cv2.putText(ventana, "Grabando... Habla ahora", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.imshow("Grabacion", ventana)
        cv2.waitKey(1)

        audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        cv2.destroyWindow("Grabacion")

        y = audio.flatten()
        try:
            features = extraer_features_desde_array(y)
            features_norm = scaler.transform([features])[0]
            pred = modelo.predict(np.expand_dims(features_norm, axis=0), verbose=0)[0]
            predicciones = sorted(zip(ETIQUETAS, pred), key=lambda x: x[1], reverse=True)

            # Mostrar resultados
            alto_ventana = 40 + len(predicciones) * 35
            ventana = 255 * np.ones((alto_ventana, 400, 3), dtype=np.uint8)
            cv2.putText(ventana, "Predicciones:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            for i, (palabra, prob) in enumerate(predicciones):
                texto = f"{palabra}: {prob:.1%}"
                cv2.putText(ventana, texto, (10, 65 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Resultados", ventana)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error al procesar: {e}")
            continue
        
    elif key == 27:  # ESC para salir
        break

cv2.destroyAllWindows()