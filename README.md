# Acoustic Emergency Vehicle Detector

This repository contains the source code for a real-time machine learning system designed to acoustically detect emergency vehicles (ambulances, police cars, fire engines) using a Raspberry Pi. 

This project was developed as part of a Master's Thesis and serves as a practical demonstration of end-to-end ML deployment on edge devices.

## Watch Demo on YouTube! 

https://www.youtube.com/watch?v=fpkjyRfE9QY

## Project Overview

The system continuously listens to the environment using an I2S MEMS microphone, processes the audio signal, and classifies it in real-time. If an emergency siren is detected, it alerts the user via an OLED display.

To ensure robustness and compare different approaches, three distinct machine learning models were developed and tested:
1. **Support Vector Machine (SVM)** - based on MFCC features.
2. **Convolutional Neural Network (CNN)** - a custom architecture trained on Log-Mel Spectrograms.
3. **YAMNet (Transfer Learning)** - utilizing Google's pre-trained audio event classifier combined with a custom dense classification head.

## Hardware Requirements
* **Raspberry Pi 4 Model B**
* **INMP441** Omnidirectional I2S MEMS Microphone
* **SSD1306** I2C OLED Display (128x32)
* Jumper wires & Breadboard

## Repository Structure

The project is divided into logical components separating model training (intended for a PC) from real-time inference (intended for the Raspberry Pi).
```text
├── rpi_deployment/        # Real-time inference scripts for Raspberry Pi
│   ├── run_cnn.py
│   ├── run_svm.py
│   └── run_yamnet.py
├── training/              # Scripts for feature extraction and model training
│   ├── train_cnn.py
│   ├── train_svm.py
│   └── train_yamnet.py
├── models/                # Directory for trained model weights and scalers
├── data/                  # Directory for audio datasets
├── results/               # Experiment logs, metrics, and confusion matrices
├── assets/                # Fonts and static assets for the OLED display
├── requirements-training.txt # Python dependencies for model training (PC)
└── requirements-rpi.txt      # Python dependencies for inference (Raspberry Pi)
```
## Quick Start

### 1. Model Training (PC / Server)
It is highly recommended to train the models on a standard PC/Mac with a dedicated GPU (if available) rather than the Raspberry Pi.

```bash
# Clone the repository
git clone https://github.com/yourusername/acoustic-emergency-detector.git
cd acoustic-emergency-detector

# Install training dependencies
pip install -r requirements-training.txt

# Prepare your dataset (see data/README.md for structure)

# Run a training script, for example:
python training/train_cnn.py
```
*Trained models will be automatically saved to the `models/` directory.*

### 2. Edge Deployment (Raspberry Pi)
Transfer the cloned repository (along with the trained models in the `models/` folder) to your Raspberry Pi.

```bash
# Install inference dependencies
pip install -r requirements-rpi.txt

# Run the real-time detection script (e.g., CNN)
python rpi_deployment/run_cnn.py
```

## System Architecture & Data Flow
1. **Audio Capture:** I2S microphone captures audio at 16 kHz.
2. **Preprocessing:** Peak normalization and bandpass filtering (300Hz - 5000Hz) to remove low-frequency rumble and high-frequency noise.
3. **Feature Extraction:** Sliding window technique generating MFCCs, Delta MFCCs, or Log-Mel Spectrograms.
4. **Inference:** The selected model predicts the probability of a siren being present in the current 3-second audio buffer.
5. **Post-processing (EMA):** Exponential Moving Average is applied to raw prediction probabilities to prevent flickering and reduce false positives.
6. **Output:** The SSD1306 OLED display updates its status (`Traffic / Silence` or `!UWAGA ALARM!`).
