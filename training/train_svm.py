import os
from pathlib import Path
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)

# UNIWERSALNE ŚCIEŻKI OPARTE O STRUKTURĘ PROJEKTU
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'Audio_Balanced_3s'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RATE = 16000
DURATION = 3
SAMPLES  = RATE * DURATION
N_MFCC   = 40
CLASSES  = ['traffic', 'alarm']

# Filtr pasmowoprzepustowy
sos = signal.butter(5, [300, 5000], 'bandpass', fs=RATE, output='sos')

def get_features(file_path):
    """
    Wczytuje plik audio i zwraca spłaszczony wektor cech (MFCC + delta + delta-delta).
    """
    try:
        audio, _ = librosa.load(file_path, sr=RATE)

        # Ujednolicenie długości
        if len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        elif len(audio) < SAMPLES:
            audio = np.pad(audio, (0, max(0, SAMPLES - len(audio))), 'constant')

        # Normalizacja
        if max(audio) - min(audio) != 0:
            audio = 2 * ((audio - min(audio)) / (max(audio) - min(audio))) - 1

        # Filtrowanie
        audio = signal.sosfilt(sos, audio)

        # Ekstrakcja
        mfccs        = librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=N_MFCC)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        feature_vector = np.concatenate([
            np.mean(mfccs,        axis=1), np.std(mfccs,        axis=1),
            np.mean(delta_mfccs,  axis=1), np.std(delta_mfccs,  axis=1),
            np.mean(delta2_mfccs, axis=1), np.std(delta2_mfccs, axis=1),
        ])
        return feature_vector
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {file_path}: {e}")
        return None

def load_data(data_dir):
    features, labels = [], []
    print("Rozpoczęto ekstrakcję cech. To może potrwać kilka minut...")
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"OSTRZEŻENIE: Brak folderu {class_dir}")
            continue

        count = 0
        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_path = class_dir / file_name
                fv = get_features(file_path)
                if fv is not None:
                    features.append(fv)
                    labels.append(label_idx)
                    count += 1
        print(f"  Klasa '{class_name}': {count} plików.")

    return np.array(features), np.array(labels)

# 1. PRZYGOTOWANIE DANYCH
X, y = load_data(DATA_DIR)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# 2. TRENOWANIE SVM
print("\nRozpoczęcie treningu SVM...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(x_train, y_train)
print("Trening zakończony.")

# 3. OCENA
y_pred = svm_model.predict(x_test)
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall    = recall_score(y_test, y_pred, average='macro')

print("\n========== WYNIKI NA ZBIORZE TESTOWYM ==========")
print(f"Accuracy:     {accuracy*100:.2f}%")
print(f"F1 (macro):   {f1*100:.2f}%")
print("=================================================")

# 4. MACIERZ POMYŁEK
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'Macierz pomyłek — SVM\nAccuracy: {accuracy*100:.2f}%')
plt.tight_layout()

cm_path = RESULTS_DIR / 'SVM_confusion_matrix.png'
plt.savefig(cm_path, dpi=150)
print(f"Macierz pomyłek zapisana: {cm_path}")

# 5. ZAPIS METRYK
metrics_path = RESULTS_DIR / 'SVM_metrics.txt'
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("========== WYNIKI NA ZBIORZE TESTOWYM — SVM ==========\n")
    f.write(f"Accuracy:     {accuracy*100:.2f}%\n")
    f.write(f"F1 (macro):   {f1*100:.2f}%\n")
    f.write(f"Precision:    {precision*100:.2f}%\n")
    f.write(f"Recall:       {recall*100:.2f}%\n")
    f.write(np.array2string(cm))

# 6. ZAPIS MODELU I SCALERA
model_path  = MODELS_DIR / 'SVM_model_NBP.joblib'
scaler_path = MODELS_DIR / 'SVM_scaler_NBP.joblib'
joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Zapisano model i scaler w {MODELS_DIR}")