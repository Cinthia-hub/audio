import os
import time
import uuid
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import csv

# Configuración por defecto (coincide con get_feactures_no_silent.py)
DEFAULT_TARGET_ROOT = "pruebas_audio"
DEFAULT_FEATURES_CSV = "features_test.csv"
DEFAULT_CLASSES = ['down', 'go', 'left', 'stop', 'up']
DEFAULT_SR = 16000
DEFAULT_DURATION = 2.0  # segundos por defecto
N_MFCC = 13

def ensure_folders(classes, root):
    os.makedirs(root, exist_ok=True)
    for c in classes:
        path = os.path.join(root, c)
        os.makedirs(path, exist_ok=True)

def record_audio(filename, duration, sr):
    print(f"Grabando {duration:.2f} s a {sr} Hz... (habla ahora)")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    # Normalizar a -1..1 si es necesario
    maxv = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if maxv > 1.0:
        audio = audio / maxv
    sf.write(filename, audio, sr)
    print(f"Guardado: {filename}")
    return filename

def extract_features_from_file(audio_path, sr=DEFAULT_SR, n_mfcc=N_MFCC):
    y, sr = librosa.load(audio_path, sr=sr)
    y, _ = librosa.effects.trim(y)

    # MFCC y estadísticas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral centroid y bandwidth
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [rms_mean, rms_std],
        [np.mean(centroid)], [np.mean(bandwidth)],
        [np.mean(zcr)],
        chroma_mean
    ])
    return features

def append_features_to_csv(features, label, csv_path):
    header = [f"feat_{i}" for i in range(len(features))] + ["label"]
    row = list(map(float, features)) + [label]

    write_header = not os.path.exists(csv_path)
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    if write_header:
        print(f"CSV creado: {csv_path} (con encabezado)")
    print(f"Fila añadida a {csv_path}: label={label}")

def load_classes_from_labels_npy(npy_path="label_classes.npy"):
    if os.path.exists(npy_path):
        try:
            arr = np.load(npy_path, allow_pickle=True)
            arr = [str(x) for x in arr]
            if len(arr) > 0:
                return arr
        except Exception:
            pass
    return None

def interactive_record_loop(classes, root, duration, sr, csv_path):
    print("=== Grabadora interactiva ===")
    print("Clases disponibles:", classes)
    print("Escribe el nombre de la clase a grabar (o 'q' para salir).")
    print("Al presionar Enter comenzarás la grabación de duración:", duration, "s")
    while True:
        clase = input("Clase (o 'q' para salir): ").strip()
        if clase.lower() == 'q':
            print("Saliendo.")
            break
        if clase not in classes:
            print(f"Clase '{clase}' no reconocida. Clases válidas: {classes}")
            continue
        # generar nombre único
        filename = f"{clase}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        filepath = os.path.join(root, clase, filename)
        try:
            record_audio(filepath, duration=duration, sr=sr)
            feats = extract_features_from_file(filepath, sr=sr, n_mfcc=N_MFCC)
            append_features_to_csv(feats, clase, csv_path=csv_path)
            print("Grabación y extracción completadas.\n")
        except Exception as e:
            print("Error durante la grabación o extracción:", e)

def non_interactive_batch(classes, root, duration, sr, csv_path, count_per_class):
    """
    Graba count_per_class archivos por cada clase automáticamente.
    """
    print(f"Modo batch: {count_per_class} grabaciones por clase en {root}/<clase>/")
    for c in classes:
        for i in range(count_per_class):
            filename = f"{c}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
            filepath = os.path.join(root, c, filename)
            try:
                record_audio(filepath, duration=duration, sr=sr)
                feats = extract_features_from_file(filepath, sr=sr, n_mfcc=N_MFCC)
                append_features_to_csv(feats, c, csv_path=csv_path)
                time.sleep(0.5)  # pequeño descanso entre grabaciones
            except Exception as e:
                print(f"Error grabando {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Crear carpetas y grabar audio para try_model2 / features_test.csv")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Duración de cada grabación (s)")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Frecuencia de muestreo (Hz)")
    parser.add_argument("--root", type=str, default=DEFAULT_TARGET_ROOT, help="Carpeta raíz para las clases")
    parser.add_argument("--csv", type=str, default=DEFAULT_FEATURES_CSV, help="CSV de features resultante")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo (pregunta por clase y graba)")
    parser.add_argument("--count", type=int, default=0, help="Si >0, número de grabaciones automáticas por clase (modo batch)")
    args = parser.parse_args()

    # Intentar leer clases desde label_classes.npy si existe; si no, usar DEFAULT_CLASSES
    classes_npy = load_classes_from_labels_npy("label_classes.npy")
    classes = classes_npy if classes_npy is not None else DEFAULT_CLASSES

    # Asegurar carpetas (usar el root pasado por argumentos)
    ensure_folders(classes, root=args.root)
    print(f"Carpeta raíz '{args.root}' y subcarpetas para clases creadas/verificadas.")

    # Variables locales derivadas de argumentos
    sr = args.sr
    duration = args.duration
    root = args.root
    csv_path = args.csv

    if args.count and args.count > 0:
        non_interactive_batch(classes, root=root, duration=duration, sr=sr, csv_path=csv_path, count_per_class=args.count)
    elif args.interactive:
        interactive_record_loop(classes, root=root, duration=duration, sr=sr, csv_path=csv_path)
    else:
        print("Modo no interactivo: carpetas creadas y disponibles para grabar.")
        print("Para grabar e insertar features en el CSV ejecuta con --interactive")
        print("Ejemplo: python create_recordings_and_features.py --interactive --duration 2 --sr 16000")
        print("O para grabar automáticamente 3 muestras por clase:")
        print("  python create_recordings_and_features.py --count 3 --duration 2")

if __name__ == "__main__":
    main()