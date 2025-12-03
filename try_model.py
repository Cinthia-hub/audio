import librosa
import numpy as np
import tensorflow as tf
import cv2
import joblib  # Para cargar el scaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import pandas as pd

# Parámetros
FS = 16000
N_MFCC = 13
ARCHIVO_AUDIO = "pruebas_audio/four/4_george_2.wav"
MODELO_PATH = "modelo_audio.keras"
SCALER_PATH = "scaler.pkl"  # Ruta al scaler entrenado
FEATURES_TEST_CSV = "features_test.csv"
LABELS_NPY = "label_classes.npy"

# Etiquetas por defecto (solo si no existe label_classes.npy)
etiquetas = ['one', 'two', 'three', 'four', 'five']

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

def save_text_report_as_figure(text, filename):
    lines = text.splitlines()
    height = max(2.0, 0.22 * len(lines) + 0.8)
    fig = plt.figure(figsize=(7.5, height))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.01, 0.99, text, fontfamily='monospace', fontsize=10, va='top', ha='left')
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved classification report figure to {filename}")

# Cargar modelo y scaler
modelo = tf.keras.models.load_model(MODELO_PATH)
scaler = joblib.load(SCALER_PATH)

# Cargar clases guardadas si existen
if os.path.exists(LABELS_NPY):
    clases = np.load(LABELS_NPY, allow_pickle=True)
    clases = np.asarray(clases, dtype=str)
    print("Clases cargadas desde", LABELS_NPY, "=>", clases)
else:
    clases = np.array(etiquetas)
    print("label_classes.npy no encontrado. Usando etiquetas definidas en el script:", clases)

encoder = LabelEncoder()
encoder.classes_ = clases

try:
    print(f" Cargando archivo: {ARCHIVO_AUDIO}")
    y, sr = librosa.load(ARCHIVO_AUDIO, sr=FS)
    features = extraer_features_desde_array(y)
    features_norm = scaler.transform([features])[0]
    pred = modelo.predict(np.expand_dims(features_norm, axis=0), verbose=0)[0]

    # Mostrar predicciones individuales
    predicciones = sorted(zip(clases, pred), key=lambda x: x[1], reverse=True)
    for palabra, prob in predicciones:
        print(f"{palabra:>7}: {prob:.2%}")

    # Mostrar en ventana OpenCV
    alto_ventana = 40 + len(predicciones) * 35
    ventana = 255 * np.ones((alto_ventana, 400, 3), dtype=np.uint8)
    cv2.putText(ventana, "Predicciones:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for i, (palabra, prob) in enumerate(predicciones):
        texto = f"{palabra}: {prob:.1%}"
        cv2.putText(ventana, texto, (10, 65 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Resultados", ventana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Si existe features_test.csv, calculamos métricas globales y guardamos figura textual
    if os.path.exists(FEATURES_TEST_CSV):
        df = pd.read_csv(FEATURES_TEST_CSV)
        if "label" in df.columns:
            print("Calculando métricas globales usando", FEATURES_TEST_CSV)
            X_test = df.drop(columns=["label"]).values.astype(np.float32)
            y_test = df["label"].values
            X_test_scaled = scaler.transform(X_test)
            y_pred_probs = modelo.predict(X_test_scaled, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            # Intentar usar las clases guardadas; si no coinciden se infiere desde y_test
            try:
                y_test_enc = encoder.transform(y_test)
                target_names = encoder.classes_
            except Exception:
                clases_inf = np.unique(y_test)
                encoder2 = LabelEncoder()
                encoder2.classes_ = np.asarray(clases_inf, dtype=str)
                y_test_enc = encoder2.transform(y_test)
                target_names = encoder2.classes_
                print("Advertencia: clases en label_classes.npy no coinciden con CSV de test. Usando clases inferidas desde CSV.")
            acc = accuracy_score(y_test_enc, y_pred)
            report = classification_report(y_test_enc, y_pred, target_names=target_names, digits=4, zero_division=0)
            text = f"Accuracy: {acc:.4f}\n\nClassification report:\n{report}"
            save_text_report_as_figure(text, "classification_report_try_model.png")
        else:
            print("El CSV de test no contiene columna 'label'.")
    else:
        print(f"No se encontró {FEATURES_TEST_CSV}; se omitió cálculo de métricas globales.")

except Exception as e:
    print(f" Error al procesar: {e}")