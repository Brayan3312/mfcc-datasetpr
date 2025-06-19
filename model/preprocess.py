import os
import csv

dataset_path = 'data/augmented'
output_csv = 'data/audio_files_custom.csv'

with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['filename', 'speaker'])

    for speaker in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith('.wav'):
                    relative_path = os.path.join(speaker, audio_file)
                    writer.writerow([relative_path, speaker])

print("✅ CSV actualizado con todos los audios, incluyendo los aumentados.")
