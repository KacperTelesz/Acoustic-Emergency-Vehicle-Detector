# Trained Models

This directory contains the serialized weights and artifacts of the trained machine learning models. The inference scripts running on the Raspberry Pi (`rpi_deployment/`) will load the models from this directory.

## Expected Files
If you are running the project from scratch, training the models will output the following files here:

* `CNN_model_LogMel.keras` - Saved Keras model for the Convolutional Neural Network.
* `SVM_model_NBP.joblib` - Scikit-Learn SVC model.
* `SVM_scaler_NBP.joblib` - Scikit-Learn StandardScaler fitted on the SVM training data (crucial for inference).
* `YamNet_V1_model_Audio_Balanced_3s.keras` - The custom dense classification head trained on top of YAMNet embeddings.

*Note: Pre-trained models are included in the repository for demonstration purposes. If you wish to retrain them, the old files will be overwritten.*