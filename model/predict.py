import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import sys

# === Configuración ===
MODEL_PATH = 'model/model_custom.h5'
ENCODER_PATH = 'model/label_encoder_custom.pkl'
AUDIO_PATH = 'data/predict/prueba100.wav'
  # <- Cambia esto por tu archivo a predecir

# === 1. Cargar el modelo y codificador de etiquetas ===
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# === 2. Función para extraer MFCC del nuevo audio ===
def extract_mfcc(file_path, max_pad_len=12996):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # Si quieres usar padding en lugar de media:
    # pad_width = max_pad_len - mfcc.shape[1]
    # if pad_width > 0:
    #     mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # else:
    #     mfcc = mfcc[:, :max_pad_len]
    # mfcc_scaled = mfcc.flatten()

    return mfcc_scaled

# === 3. Extraer MFCC del nuevo audio ===
mfcc_features = extract_mfcc(AUDIO_PATH)
mfcc_features = mfcc_features.reshape(1, -1)  # Adaptar dimensión para el modelo

# === 4. Predecir ===
prediction = model.predict(mfcc_features)
predicted_index = np.argmax(prediction)
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

# === 5. Resultado ===
print(f"🔊 El locutor predicho es: {predicted_label}")
