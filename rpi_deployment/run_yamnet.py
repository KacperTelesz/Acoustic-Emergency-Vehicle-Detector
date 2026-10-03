import sys
import types
import importlib.util
from pathlib import Path

fake_imp = types.ModuleType('imp')
def mock_find_module(name, *args):
    if importlib.util.find_spec(name) is None:
        raise ImportError(name)
    return (None, None, None)
fake_imp.find_module = mock_find_module
sys.modules['imp'] = fake_imp

import numpy as np
import pyaudio
import tensorflow as tf
import tensorflow_hub as hub
import time
import csv
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

# KONFIGURACJA ŚCIEŻEK (DYNAMICZNA)
BASE_DIR = Path(__file__).resolve().parent.parent
YAMNET_PATH = 'https://tfhub.dev/google/yamnet/1'
MODEL_PATH = BASE_DIR / 'models' / 'YamNet_model_Claude_Audio_Balanced_3s.keras'
ASSETS_DIR = BASE_DIR / 'assets'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETECTION_MODE = 'classifier' # lub 'yamnet_direct'
YAMNET_SIREN_INDICES = [388, 389, 390, 391, 396]
FRAME_THRESH = 0.40
PROB_THRESH = 0.80
EMA_ALPHA = 0.6
ema_score = 0.0
alarm_active = False

print("Ladowanie YamNet z TensorFlow Hub...")
yamnet_model = hub.load(YAMNET_PATH)

if DETECTION_MODE == 'classifier':
    print(f"Budowanie szkieletu modelu i wczytywanie wag z {MODEL_PATH}...")
    CLASSES = ['traffic', 'alarm']
    ALARM_IDX = CLASSES.index('alarm')
    
    classifier_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    try:
        classifier_model.load_weights(str(MODEL_PATH))
    except Exception as e:
        print(f"Ostrzeżenie: Nie wczytano wag modelu {e}")
else:
    classifier_model = None
    CLASSES = ['traffic', 'alarm']
    ALARM_IDX = 1
    
SCENARIO_NAME = "Scenariusz_YamNet_log"
CSV_FILENAME = RESULTS_DIR / f"{SCENARIO_NAME}.csv"
ENABLE_CSV_LOGGING = False

if ENABLE_CSV_LOGGING:
    csv_file = open(CSV_FILENAME, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time_s", "Raw_Prob", "EMA_Prob", "Predicted_Label"])
else:
    csv_file = None
    csv_writer = None
    
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    disp = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c)
    disp.fill(0)
    disp.show()
    width = disp.width
    height = disp.height
except Exception as e:
    width, height = 128, 32
    disp = None
    
image = Image.new("1", (width, height))
draw = ImageDraw.Draw(image)

try:
    font_small = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans.ttf"), 9)
    font_large = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans-Bold.ttf"), 12)
except IOError:
    font_small = ImageFont.load_default()
    font_large = ImageFont.load_default()
    
def update_display(label, p_alarm):
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    if label == 1:
        draw.rectangle((0, 0, width, height), outline=255, fill=255)
        draw.text((10, 2), "!UWAGA ALARM!", font=font_large, fill=0)
        draw.text((10, 18), "  Zachowaj Ostroznosc", font=font_small, fill=0)
    else:
        draw.text((0, 0), "Nasluchiwanie...", font=font_small, fill=255)
        draw.text((0, 11), "Brak zagrozenia", font=font_small, fill=255)
        draw.text((0, 22), f"EMA: {p_alarm*100:.1f}%", font=font_small, fill=255)
    if disp:
        disp.image(image)
        disp.show()
        
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Blad mikrofonu: {e}")
    exit()
    
DURATION = 3
BUFFER_SIZE = int(RATE * DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def process_audio_and_predict(audio_data, current_time):
    global ema_score, alarm_active
    try:
        peak = np.max(np.abs(audio_data))
        audio_normalized = audio_data / peak if peak > 0 else audio_data
        audio_f32 = audio_normalized.astype(np.float32)
        
        if DETECTION_MODE == 'yamnet_direct':
            scores, _, _ = yamnet_model(audio_f32)
            scores_np = scores.numpy()
            per_frame_siren = np.max(scores_np[:, YAMNET_SIREN_INDICES], axis=1)
            raw_prob = float(np.mean(per_frame_siren > FRAME_THRESH))
        else:
            _, embeddings, _ = yamnet_model(audio_f32)
            emb_mean = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)
            predictions = classifier_model.predict(emb_mean, verbose=0)
            raw_prob = float(predictions[0][ALARM_IDX])
            
        ema_score = EMA_ALPHA * raw_prob + (1 - EMA_ALPHA) * ema_score
        predicted_label = 'alarm' if ema_score > PROB_THRESH else 'traffic'
        
        if ENABLE_CSV_LOGGING:
            csv_writer.writerow([f"{current_time:.3f}", f"{raw_prob:.4f}", f"{ema_score:.4f}", predicted_label])
        update_display(1 if predicted_label == 'alarm' else 0, ema_score)
        
        if predicted_label == 'alarm':
            if not alarm_active:
                alarm_active = True
            print(f"[{current_time:.1f}s] UWAGA: WYKRYTO SYRENE! (EMA: {ema_score*100:.1f}%, Surowe: {raw_prob*100:.1f}%)")
        else:
            if alarm_active:
                alarm_active = False
            print(f"[{current_time:.1f}s] Ruch uliczny / Cisza... (EMA: {ema_score*100:.1f}%, Surowe: {raw_prob*100:.1f}%)")
    except Exception as e:
        print(f"Wystapil blad predykcji: {e}")
        
print(f"Rozpoczeto nasluch w trybie '{DETECTION_MODE}'... Nacisnij Ctrl+C, aby zatrzymac.")
start_time = time.time()
CHUNKS_PER_PREDICTION = 7
chunks_read = 0

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_int16 = np.frombuffer(data, dtype=np.int16)
        data_float = data_int16.astype(np.float32) / 32768.0
        
        audio_buffer = np.roll(audio_buffer, -CHUNK)
        audio_buffer[-CHUNK:] = data_float
        chunks_read += 1
        
        if chunks_read >= CHUNKS_PER_PREDICTION:
            elapsed_time = time.time() - start_time
            process_audio_and_predict(audio_buffer, elapsed_time)
            chunks_read = 0
except KeyboardInterrupt:
    print("
Otrzymano sygnal zatrzymania...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    if ENABLE_CSV_LOGGING and csv_file:
        csv_file.close()
    update_display(0, 0.0)
    if disp:
        disp.fill(0)
        disp.show()
