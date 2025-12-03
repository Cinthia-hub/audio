import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Parámetros rápidos para prueba
EPOCHS = 10        # bajar para entrenar rápido; aumenta si quieres mejor rendimiento
BATCH_SIZE = 32

# Archivo de features (asegúrate que existe features.csv)
CSV_PATH = "features.csv"
MODEL_KERAS = "modelo_audio.keras"
MODEL_SAVED = "modelo_audio"  # carpeta SavedModel (opcional)
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "label_classes.npy"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No se encontró {CSV_PATH} en el directorio actual.")

# 1. Cargar datos
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# 2. Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler guardado en {SCALER_PATH}")

# 3. Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
np.save(LABELS_PATH, encoder.classes_)
print(f"Clases guardadas en {LABELS_PATH}: {encoder.classes_}")

num_classes = len(encoder.classes_)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Modelo simple
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Entrenar (rápido)
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=2)

# 7. Evaluar y guardar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy (rápido): {acc:.4f}")

# Guardar en formato .keras (zip)
model.save(MODEL_KERAS)
print(f"Modelo guardado en {MODEL_KERAS}")

# Guardar como SavedModel (carpeta) usando tf.saved_model.save
try:
    tf.saved_model.save(model, MODEL_SAVED)
    print(f"Modelo guardado en carpeta {MODEL_SAVED} (SavedModel format)")
except Exception as e:
    print(f"No se pudo guardar como SavedModel: {e}")
    print("Esto no es crítico si ya tienes el archivo .keras.")