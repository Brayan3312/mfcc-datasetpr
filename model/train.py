import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# === 1. Cargar datos
data = np.load('data/mfcc_custom_dataset.npz')
mfccs = data['mfccs']
labels = data['labels']

# === 2. Codificar etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# === 3. Dividir datos
X_train, X_val, y_train, y_val = train_test_split(
    mfccs, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
)

# === 4. Crear modelo
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(mfccs.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(labels_categorical.shape[1], activation='softmax'))

# === 5. Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 6. Callback
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === 7. Entrenar
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# === 8. Guardar modelo y codificador
os.makedirs('model', exist_ok=True)
model.save('model/model_custom.h5')
with open('model/label_encoder_custom.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Entrenamiento completo. Modelo y codificador guardados.")

# === 9. Evaluación profunda

# Convertir etiquetas codificadas one-hot a clases
y_true_classes = np.argmax(y_val, axis=1)
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# === Matriz de confusión
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45, cmap="Blues")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# === Reporte de clasificación
report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_)
print("📋 Reporte de Clasificación:\n")
print(report)
