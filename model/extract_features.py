import os
import librosa
import numpy as np
import tqdm
import soundfile as sf

# Ruta base donde están las carpetas por locutor
DATASET_PATH = "data/augmented"
OUTPUT_PATH = "data/mfcc_custom_dataset.npz"

mfccs = []
labels = []

# Recorrer cada carpeta (locutor)
for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    if not os.path.isdir(speaker_path):
        continue

    print(f"🔍 Procesando locutor: {speaker}")

    # Recorrer cada archivo de audio
    for file in tqdm.tqdm(os.listdir(speaker_path)):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(speaker_path, file)

        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)  # Extraer características promedio

            mfccs.append(mfcc_mean)
            labels.append(speaker)

        except Exception as e:
            print(f"❌ Error en {file_path}: {e}")

# Guardar MFCCs y etiquetas
np.savez(OUTPUT_PATH, mfccs=np.array(mfccs), labels=np.array(labels))
print(f"✅ MFCCs guardados en {OUTPUT_PATH}")
