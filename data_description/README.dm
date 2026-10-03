# Dataset Directory

Due to GitHub file size limits and privacy concerns, the raw audio dataset is not included in this repository.

To train the models using the scripts in the `training/` directory, please place your audio recordings here following the exact structure below.

## Required Structure

The training scripts expect a folder named `Audio_Balanced_3s` inside this directory, with subfolders representing the classes:

```text
data/
└── Audio_Balanced_3s/
    ├── alarm/
    │   ├── ambulance_01.wav
    │   ├── police_02.mp3
    │   └── fire_truck_03.flac
    └── traffic/
        ├── street_noise_01.wav
        ├── silence_02.wav
        └── wind_03.mp3
```

## Audio Specifications
* The models were designed for **16 kHz** sample rate. (The scripts will automatically resample if necessary, but providing 16 kHz files is faster).
* Expected audio length is **3 seconds**. Audio will be padded or truncated automatically by the feature extraction pipeline.
