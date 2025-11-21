import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Lista de archivos de audio
archivos = ["prueba1.ogg", "prueba2.ogg"]

# Crear carpeta de salida
os.makedirs("espectrogramas", exist_ok=True)

# Procesar cada archivo
for audio_path in archivos:
    print(f"Procesando {audio_path}...")
    
    # --- 1. Cargar audio en mono a 16 kHz ---
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # --- 2. Generar espectrograma Mel ---
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # --- 3. Generar espectrograma log-Mel ---
    S_logmel = librosa.power_to_db(S_mel, ref=np.max)
    
    # --- 4. Visualizar y guardar espectrograma Mel ---
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_mel, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.title(f"Espectrograma Mel - {audio_path}")
    plt.colorbar(format='%+2.0f')
    plt.tight_layout()
    plt.savefig(f"espectrogramas/{os.path.splitext(audio_path)[0]}_mel.png", dpi=150)
    plt.close()

    # --- 5. Visualizar y guardar espectrograma log-Mel ---
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_logmel, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.title(f"Espectrograma Log-Mel - {audio_path}")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"espectrogramas/{os.path.splitext(audio_path)[0]}_logmel.png", dpi=150)
    plt.close()

print("✅ Espectrogramas generados en la carpeta 'espectrogramas'")
