import os
from pathlib import Path
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# UNIWERSALNE ŚCIEŻKI
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'Audio_Balanced_3s'
MODELS_DIR = BASE_DIR / 'models'
CACHE_DIR = BASE_DIR / '.tfhub_cache'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ustawienie lokalizacji cache dla modeli z TF Hub (żeby nie śmieciło w systemie)
os.environ['TFHUB_CACHE_DIR'] = str(CACHE_DIR)

RATE = 16000
DURATION = 3
SAMPLES = RATE * DURATION
CLASSES = ['traffic', 'alarm'] 

print("Ładowanie YamNet z TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YamNet załadowany.")

def get_embeddings(file_path):
    """Wczytuje plik audio i wyciąga embedding YamNet (1024-dim)."""
    try:
        audio, _ = librosa.load(file_path, sr=RATE)
        if len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        elif len(audio) < SAMPLES:
            audio = np.pad(audio, (0, max(0, SAMPLES - len(audio))), 'constant')

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        audio = audio.astype(np.float32)
        _, embeddings, _ = yamnet_model(audio)

        # Uśrednienie po ramkach
        return np.mean(embeddings.numpy(), axis=0)

    except Exception as e:
        print(f"Błąd przetwarzania {file_path}: {e}")
        return None

def load_data(data_dir):
    features, labels = [], []
    print("Rozpoczęto ekstrakcję cech YamNet...")
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Brak folderu {class_dir}")
            continue

        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_path = class_dir / file_name
                embedding = get_embeddings(file_path)
                if embedding is not None:
                    features.append(embedding)
                    labels.append(label_idx)

    return np.array(features), np.array(labels)

# 1. PRZYGOTOWANIE DANYCH
X, y = load_data(DATA_DIR)
y = to_categorical(y, num_classes=len(CLASSES))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODEL
model = Sequential([
    Dense(256, activation='relu', input_shape=(1024,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. CALLBACKS
model_save_path = str(MODELS_DIR / 'YamNet_V1_model_Audio_Balanced_3s.keras')
callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# 4. TRENOWANIE
print("Rozpoczęcie treningu...")
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Dokładność na zbiorze testowym: {score[1]*100:.2f}%")