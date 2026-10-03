import sys
import types
import importlib.util
from pathlib import Path

# Atrapa modułu 'imp' dla kompatybilności z nowszymi wersjami Pythona
fake_imp = types.ModuleType('imp')
def mock_find_module(name, *args):
    if importlib.util.find_spec(name) is None:
        raise ImportError(name)
    return (None, None, None)
fake_imp.find_module = mock_find_module
sys.modules['imp'] = fake_imp

import numpy as np
import pyaudio
import librosa
import tensorflow as tf
import keras
from scipy import signal
import time
import csv
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

# Globalny Monkey Patch dla BatchNormalization (Keras 2 -> Keras 3)
original_bn_init = keras.layers.BatchNormalization.__init__

def patched_bn_init(self, *args, **kwargs):
    keras2_only_args = ('renorm', 'renorm_clipping', 'renorm_momentum')
    for key in keras2_only_args:
        kwargs.pop(key, None)
    original_bn_init(self, *args, **kwargs)

keras.layers.BatchNormalization.__init__ = patched_bn_init
tf.keras.layers.BatchNormalization.__init__ = patched_bn_init

# KONFIGURACJA ŚCIEŻEK (DYNAMICZNA)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'CNN_model_LogMel.keras'
ASSETS_DIR = BASE_DIR / 'assets'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PARAMETRY I ŁADOWANIE MODELU
print(f"Ladowanie modelu z {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
except Exception as e:
    print(f"Blad ladowania modelu: {e}")
    exit()

TARGET_RATE = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
F_MIN = 300
F_MAX = 5000
EXPECTED_FRAMES = 301
CLASSES = ['traffic', 'alarm']

USE_RESAMPLING = False
PROB_THRESH = 0.80
EMA_ALPHA = 0.7
ema_score = 0.0
alarm_active = False

sos = signal.butter(5, [300, 5000], 'bandpass', fs=TARGET_RATE, output='sos')

SCENARIO_NAME = "Scenariusz_1_Pokoj"
CSV_FILENAME = RESULTS_DIR / f"{SCENARIO_NAME}_CNN_log.csv"
ENABLE_CSV_LOGGING = False

if ENABLE_CSV_LOGGING:
    csv_file = open(CSV_FILENAME, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time_s", "Raw_Prob", "EMA_Prob", "Predicted_Label"])
    print(f"Dane beda logowane do pliku: {CSV_FILENAME}")
else:
    csv_file = None
    csv_writer = None
    print("Logowanie do CSV wylaczone.")

# KONFIGURACJA OLED
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    disp = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c)
    disp.fill(0)
    disp.show()
    width = disp.width
    height = disp.height
except Exception as e:
    print(f"OSTRZEZENIE: Brak ekranu I2C: {e}")
    width, height = 128, 32
    disp = None

image = Image.new("1", (width, height))
draw = ImageDraw.Draw(image)

try:
    font_small = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans.ttf"), 9)
    font_large = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans-Bold.ttf"), 12)
except IOError:
    print("OSTRZEZENIE: Nie znaleziono pliku .ttf! Uzywam domyslnej czcionki.")
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

# MIKROFON
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_MIC = 16000
p = pyaudio.PyAudio()

print("Proba uruchomienia mikrofonu...")
try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE_MIC, input=True, frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Blad mikrofonu: {e}")
    exit()

DURATION = 3
BUFFER_SIZE = int(RATE_MIC * DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def process_audio_and_predict(audio_data, current_time):
    global ema_score, alarm_active
    try:
        if USE_RESAMPLING and RATE_MIC != TARGET_RATE:
            audio_processed = librosa.resample(y=audio_data, orig_sr=RATE_MIC, target_sr=TARGET_RATE)
        else:
            audio_processed = audio_data
        
        peak = np.max(np.abs(audio_processed))
        if peak > 0:
            audio_processed = audio_processed / peak
            
        audio_filtered = signal.sosfilt(sos, audio_processed)
        mel_spec = librosa.feature.melspectrogram(y=audio_filtered, sr=TARGET_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        if log_mel.shape[1] < EXPECTED_FRAMES:
            log_mel = np.pad(log_mel, ((0, 0), (0, EXPECTED_FRAMES - log_mel.shape[1])), mode='constant')
        else:
            log_mel = log_mel[:, :EXPECTED_FRAMES]
            
        prediction_feature = log_mel.reshape(1, N_MELS, EXPECTED_FRAMES, 1)
        predictions = model.predict(prediction_feature, verbose=0)
        raw_prob = float(predictions[0][CLASSES.index('alarm')])
        
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
        print(f"Wystapil blad podczas predykcji: {e}")

print("Rozpoczeto nasluch... Nacisnij Ctrl+C, aby zatrzymac.")
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
    print("Wylaczono poprawnie.")
