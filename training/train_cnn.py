import os
from pathlib import Path
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, SeparableConv2D, MaxPooling2D,
                                     GlobalAveragePooling2D, Dense, Dropout,
                                     BatchNormalization)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy import signal

# UNIWERSALNE ŚCIEŻKI
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'Audio_Balanced_3s'
MODELS_DIR = BASE_DIR / 'models'

# Utworzenie folderu na modele jeśli nie istnieje
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# PARAMETRY
RATE     = 16000
DURATION = 3
SAMPLES  = RATE * DURATION   # 48 000 próbek

# Parametry log-mel
N_MELS     = 64    # liczba filtrów mel
N_FFT      = 400   # rozmiar okna FFT: 25 ms @ 16 kHz  
HOP_LENGTH = 160   # krok okna:        10 ms @ 16 kHz  
F_MIN      = 300   # dolna granica pasma 
F_MAX      = 5000  # górna granica pasma 

# Liczba ramek czasowych dla 3 s sygnału przy HOP_LENGTH=160 i center=True:
EXPECTED_FRAMES = 1 + int(np.floor(SAMPLES / HOP_LENGTH))
CLASSES  = ['traffic', 'alarm']

print(f"Parametry wejściowe CNN: ({N_MELS} mel × {EXPECTED_FRAMES} ramek)")

# Filtr pasmowoprzepustowy 300–5000 Hz
sos = signal.butter(5, [300, 5000], 'bandpass', fs=RATE, output='sos')

def get_log_mel(file_path):
    """
    Wczytuje plik audio i zwraca log-mel spectrogram o kształcie (N_MELS, EXPECTED_FRAMES).
    """
    try:
        audio, _ = librosa.load(file_path, sr=RATE)

        # Ujednolicenie długości
        if len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        elif len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)), mode='constant')

        # Normalizacja
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # Filtrowanie
        audio = signal.sosfilt(sos, audio)

        # Mel spectrogram -> skala dB
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Wyrównywanie kształtu
        if log_mel.shape[1] < EXPECTED_FRAMES:
            log_mel = np.pad(log_mel,
                             ((0, 0), (0, EXPECTED_FRAMES - log_mel.shape[1])),
                             mode='constant')
        else:
            log_mel = log_mel[:, :EXPECTED_FRAMES]

        return log_mel

    except Exception as e:
        print(f"Błąd podczas przetwarzania {file_path}: {e}")
        return None

def load_data(data_dir):
    features, labels = [], []
    print("Rozpoczęto ekstrakcję cech. To może potrwać kilka minut...")

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"OSTRZEŻENIE: Brak folderu {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        print(f"  Klasa '{class_name}': {len(files)} plików...")

        for file_name in files:
            file_path = class_dir / file_name
            log_mel   = get_log_mel(file_path)
            if log_mel is not None:
                features.append(log_mel)
                labels.append(label_idx)

    print(f"Załadowano łącznie {len(features)} plików.")
    return np.array(features), np.array(labels)

# 1. PRZYGOTOWANIE DANYCH
X, y = load_data(DATA_DIR)

# Dodanie wymiaru kanału
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print(f"Kształt zbioru danych: {X.shape}")

# One-hot encoding etykiet
y = to_categorical(y, num_classes=len(CLASSES))

# Podział 80/20
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Trening: {len(x_train)} próbek | Test: {len(x_test)} próbek")

# 2. ARCHITEKTURA MODELU
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
           input_shape=(N_MELS, EXPECTED_FRAMES, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    SeparableConv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    SeparableConv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 3. CALLBACKS
model_save_path = str(MODELS_DIR / 'CNN_model_LogMel.keras')
callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

# 4. TRENING
print("Rozpoczęcie treningu...")
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 5. OCENA
score = model.evaluate(x_test, y_test, verbose=0)
print(f"\nDokładność na zbiorze testowym: {score[1]*100:.2f}%")
print(f"Strata na zbiorze testowym:     {score[0]:.4f}")
print(f"\nModel zapisany w katalogu models/.")