import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Carpetas
input_base = 'data/dataset_original_amplificado'  # Aquí están las carpetas por locutor
output_base = 'data/augmented'  # Aquí se guardarán los nuevos audios

# Crear estructura de carpetas de salida
os.makedirs(output_base, exist_ok=True)
for speaker in os.listdir(input_base):
    os.makedirs(os.path.join(output_base, speaker), exist_ok=True)

# Aumentar cada audio con varias técnicas
def augment_audio(file_path, speaker_folder, filename):
    y, sr = librosa.load(file_path, sr=None)

    # Guardar audio original
    sf.write(os.path.join(speaker_folder, f'original_{filename}'), y, sr)

    # 1. Cambio de velocidad
    y_speed = librosa.effects.time_stretch(y, rate=1.1)
    sf.write(os.path.join(speaker_folder, f'speed_{filename}'), y_speed, sr)

    # 2. Cambio de pitch 
    y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    sf.write(os.path.join(speaker_folder, f'pitch_{filename}'), y_pitch, sr)

    # 3. Ruido
    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    sf.write(os.path.join(speaker_folder, f'noise_{filename}'), y_noise, sr)

    # 4. Shifting
    shift = np.roll(y, 1600)
    sf.write(os.path.join(speaker_folder, f'shift_{filename}'), shift, sr)

# Recorrer todos los audios
for speaker in tqdm(os.listdir(input_base)):
    speaker_path = os.path.join(input_base, speaker)
    output_speaker_path = os.path.join(output_base, speaker)

    for filename in os.listdir(speaker_path):
        if filename.endswith('.wav'):
            input_file = os.path.join(speaker_path, filename)
            augment_audio(input_file, output_speaker_path, filename)
