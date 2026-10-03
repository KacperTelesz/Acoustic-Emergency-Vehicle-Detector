import numpy as np
import pyaudio
import librosa
import joblib
import time
import csv
from scipy import signal
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# KONFIGURACJA ŚCIEŻEK (DYNAMICZNA)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'SVM_model_NBP.joblib'
SCALER_PATH = BASE_DIR / 'models' / 'SVM_scaler_NBP.joblib'
ASSETS_DIR = BASE_DIR / 'assets'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Ladowanie modelu SVM z {MODEL_PATH}...")
try:
    svm_model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    print("Model i scaler zaladowane.")
except Exception as e:
    print(f"Blad ladowania modelu SVM: {e}")
    exit()

TARGET_RATE = 16000
DURATION = 3
N_MFCC = 40
CLASSES = ['traffic', 'alarm']

USE_RESAMPLING = False
PROB_THRESH = 0.80
EMA_ALPHA = 0.7
ema_score = 0.0
alarm_active = False

sos = signal.butter(5, [300, 5000], 'bandpass', fs=TARGET_RATE, output='sos')

SCENARIO_NAME = "Scenariusz_1_Pokoj"
CSV_FILENAME = RESULTS_DIR / f"{SCENARIO_NAME}_SVM_log.csv"
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
    print(f"Brak ekranu OLED: {e}")
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
RATE_MIC = 16000
p = pyaudio.PyAudio()

try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE_MIC, input=True, frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Blad mikrofonu: {e}")
    exit()
    
BUFFER_SIZE = int(RATE_MIC * DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def get_features_realtime(audio_data):
    if USE_RESAMPLING and RATE_MIC != TARGET_RATE:
        audio = librosa.resample(y=audio_data, orig_sr=RATE_MIC, target_sr=TARGET_RATE)
    else:
        audio = audio_data
        
    if max(audio) - min(audio) != 0:
        audio = 2 * ((audio - min(audio)) / (max(audio) - min(audio))) - 1
        
    audio = signal.sosfilt(sos, audio)
    mfccs = librosa.feature.mfcc(y=audio, sr=TARGET_RATE, n_mfcc=N_MFCC)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    feature_vector = np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(delta_mfccs, axis=1), np.std(delta_mfccs, axis=1),
        np.mean(delta2_mfccs, axis=1), np.std(delta2_mfccs, axis=1),
    ])
    return feature_vector

def process_audio_and_predict(audio_data, current_time):
    global ema_score, alarm_active
    try:
        fv = get_features_realtime(audio_data)
        fv_scaled = scaler.transform(fv.reshape(1, -1))
        probs = svm_model.predict_proba(fv_scaled)[0]
        raw_prob = probs[CLASSES.index('alarm')]
        
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

print("Nasluch w toku... Nacisnij Ctrl+C aby zakonczyc.")
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
Zakonczono nasluch.")
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
