from pydub import AudioSegment
import os

carpeta_raiz = "data/dataset_original"  # Carpeta principal original
ganancia_db = 20  # Decibelios a aumentar

# Carpeta de salida con el mismo nombre + "_copia"
carpeta_salida_raiz = carpeta_raiz + "_amplificado"

# Recorre todos los directorios y subdirectorios
for root, dirs, files in os.walk(carpeta_raiz):
    for archivo in files:
        if archivo.endswith(".wav") or archivo.endswith(".mp3"):
            ruta_entrada = os.path.join(root, archivo)

            # Cargar el audio
            audio = AudioSegment.from_file(ruta_entrada)

            # Aplicar ganancia
            audio_amplificado = audio + ganancia_db

            # Ruta relativa desde carpeta_raiz
            ruta_relativa = os.path.relpath(root, carpeta_raiz)

            # Ruta de salida manteniendo estructura
            nuevo_directorio = os.path.join(carpeta_salida_raiz, ruta_relativa)
            os.makedirs(nuevo_directorio, exist_ok=True)

            # Guardar con el mismo nombre
            ruta_salida = os.path.join(nuevo_directorio, archivo)

            # Exportar el archivo
            audio_amplificado.export(ruta_salida, format="wav")
            print(f"✅ Guardado: {ruta_salida}")
