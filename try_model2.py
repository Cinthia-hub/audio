import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

# 1. Cargar los datos de test
CSV = "features_test.csv"
df = pd.read_csv(CSV)
X_test = df.drop(columns=["label"]).values.astype(np.float32)
y_test = df["label"].values

# 2. Cargar scaler, modelo y clases
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

model = tf.keras.models.load_model("modelo_audio.keras")

LABELS_NPY = "label_classes.npy"
if os.path.exists(LABELS_NPY):
    classes = np.load(LABELS_NPY, allow_pickle=True)
    classes = np.asarray(classes, dtype=str)
    print("Clases cargadas desde", LABELS_NPY, "=>", classes)
else:
    classes = np.unique(y_test)
    print("Usando clases inferidas desde CSV de test:", classes)

# 3. Codificar etiquetas verdaderas
encoder = LabelEncoder()
encoder.classes_ = classes
try:
    y_test_encoded = encoder.transform(y_test)
except Exception as e:
    # Si no coincide el conjunto de clases guardadas con las del CSV, inferir desde y_test
    clases_inf = np.unique(y_test)
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(clases_inf, dtype=str)
    y_test_encoded = encoder.transform(y_test)
    classes = encoder.classes_
    print("Advertencia: clases inferidas desde CSV de test:", classes)

# 4. Predecir
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# 5. Matriz de confusión (figura)
cm = confusion_matrix(y_test_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Matriz de Confusion (Test)")
plt.tight_layout()
plt.savefig("matriz_confusion_test.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Saved confusion matrix to matriz_confusion_test.png")

# ---- Generar figura textual con Accuracy + classification_report ----
acc = accuracy_score(y_test_encoded, y_pred)
report = classification_report(y_test_encoded, y_pred, target_names=classes, digits=4, zero_division=0)

text = f"Accuracy: {acc:.4f}\n\nClassification report:\n{report}"

# Guardar texto como figura (monospace)
lines = text.splitlines()
height = max(2.0, 0.22 * len(lines) + 0.8)
fig = plt.figure(figsize=(8.5, height))
ax = fig.add_subplot(111)
ax.axis('off')
ax.text(0.01, 0.99, text, fontfamily='monospace', fontsize=10, va='top', ha='left')
plt.tight_layout()
fig.savefig("classification_report_try_model2.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Saved classification report figure to classification_report_try_model2.png")