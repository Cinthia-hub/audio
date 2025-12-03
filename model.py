# Step 1: Load data from the CSV file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import os

# 1. Load the dataset
df = pd.read_csv("features_test.csv")  # Load extracted features and labels

# 2. Separate features and labels
X = df.drop(columns=["label"]).values.astype(np.float32)  # Features (numerical data)
y = df["label"].values 

# 3. Normalize the features
scaler = StandardScaler()  # Initialize the standard scaler
X = scaler.fit_transform(X)  # Normalize features to have mean=0 and std=1

# Save the scaler for future inference
joblib.dump(scaler, "scaler.pkl")

# 4. Encode text labels into integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # Convert string labels to integers
num_classes = len(encoder.classes_)  # Number of unique labels

print("encoder.classes_:", encoder.classes_)  # Display the classes for reference

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
# 80% for training, 20% for testing. Stratified to preserve label proportions.

# 6. Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),  # Input layer
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use integer labels
              metrics=['accuracy'])

# 7. Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)
# Train for 100 epochs, with 20% of training data used for validation

# 8. Evaluate the model on the test set
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# Save the trained model and label encoder for future use
model.save("modelo_audio.keras")
np.save("label_classes.npy", encoder.classes_)

# 9. Predict on the test set
y_pred_probs = model.predict(X_test)  # Probabilities for each class
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to predicted class index

# 10. Compute metrics and prepare reports
acc_test = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=encoder.classes_, digits=4, zero_division=0)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Save textual reports
with open("classification_report_model.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc_test:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
print("Saved textual classification report to classification_report_model.txt")

with open("metrics_summary_model.txt", "w", encoding="utf-8") as f:
    f.write("SUMMARY METRICS\n")
    f.write(f"Accuracy: {acc_test:.4f}\n\n")
    f.write("Macro averages:\n")
    f.write(f"  Precision (macro): {precision_macro:.4f}\n")
    f.write(f"  Recall    (macro): {recall_macro:.4f}\n")
    f.write(f"  F1        (macro): {f1_macro:.4f}\n\n")
    f.write("Weighted averages:\n")
    f.write(f"  Precision (weighted): {precision_weighted:.4f}\n")
    f.write(f"  Recall    (weighted): {recall_weighted:.4f}\n")
    f.write(f"  F1        (weighted): {f1_weighted:.4f}\n\n")
    f.write("Confusion matrix (counts):\n")
    np.savetxt(f, cm, fmt="%d", delimiter=",")
print("Saved metrics summary to metrics_summary_model.txt")

# 11. Create a single combined figure: Confusion matrix (left) + textual report (right)
lines = ["Accuracy: {:.4f}".format(acc_test), "", "Classification report:"] + report.splitlines()
n_lines = len(lines)
height = max(4.5, 0.35 * n_lines + 1.5)

fig = plt.figure(figsize=(12, height))
ax_conf = plt.subplot2grid((1, 3), (0, 0), colspan=2)  # left 2/3
ax_text = plt.subplot2grid((1, 3), (0, 2))            # right 1/3

# Confusion matrix plot on the left axis
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(ax=ax_conf, cmap="Blues", xticks_rotation=45, colorbar=False)
ax_conf.set_title("Confusion Matrix (Test)")

# Textual report on the right axis
ax_text.axis('off')
y0 = 0.98
dy = 0.035
for i, line in enumerate(lines):
    ax_text.text(0.01, y0 - i * dy, line, fontfamily='monospace', fontsize=9, va='top', ha='left')
ax_text.set_title("Summary")

plt.tight_layout()
out_fname = "single_report_model.png"
fig.savefig(out_fname, dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print(f"Saved combined report figure to {out_fname}")