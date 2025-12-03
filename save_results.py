import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Config
FEATURES_TEST_CSV = "features_test.csv"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "modelo_audio.keras"  # asegúrate que existe
LABELS_NPY = "label_classes.npy"   # si no existe, se usa inferencia desde test labels

# Salidas
CM_PNG = "matriz_confusion_test.png"
REPORT_TXT = "classification_report.txt"
SUMMARY_TXT = "metrics_summary.txt"

# Comprobaciones
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"No se encontró el scaler en {SCALER_PATH}")
if not os.path.exists(FEATURES_TEST_CSV):
    raise FileNotFoundError(f"No se encontró {FEATURES_TEST_CSV}")

# Cargar
print("Cargando test CSV...")
df = pd.read_csv(FEATURES_TEST_CSV)
X_test = df.drop(columns=["label"]).values.astype(np.float32)
y_test = df["label"].values

print("Cargando scaler y modelo...")
scaler = joblib.load(SCALER_PATH)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.models.load_model(MODEL_PATH)

# Cargar clases
if os.path.exists(LABELS_NPY):
    classes = np.load(LABELS_NPY, allow_pickle=True)
    print("Clases cargadas de", LABELS_NPY, "=>", classes)
else:
    classes = np.unique(y_test)
    print("LABELS_NPY no encontrado, usando clases desde y_test:", classes)

# Encode y_true
encoder = LabelEncoder()
encoder.classes_ = classes
y_test_enc = encoder.transform(y_test)

# Predecir
print("Generando predicciones...")
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matriz de confusión y reporte
cm = confusion_matrix(y_test_enc, y_pred)
acc = accuracy_score(y_test_enc, y_pred)
report = classification_report(y_test_enc, y_pred, target_names=encoder.classes_, digits=4)

# Guardar matriz como PNG
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Matriz de Confusion (Test)")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=150)
plt.close()
print(f"Matriz de confusión guardada en {CM_PNG}")

# Guardar reporte y resumen
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("Accuracy: {:.4f}\n\n".format(acc))
    f.write("Classification report:\n")
    f.write(report)
print(f"Reporte de clasificación guardado en {REPORT_TXT}")

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write("Confusion matrix (counts):\n")
    np.savetxt(f, cm, fmt="%d", delimiter=",")
print(f"Resumen de métricas guardado en {SUMMARY_TXT}")

print("Hecho.")