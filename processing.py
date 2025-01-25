import os
import pandas as pd
import librosa
import numpy as np
from joblib import load

# Load models and preprocessing tools
rf_model = load('models/random_forest_model.pkl')
knn_model = load('models/knn_model.pkl')
xgb_model = load('models/xgb_model.pkl')
scaler = load('models/scaler.pkl')
pca_model = load('models/pca_model.pkl')
label_encoder = load('models/label_encoder.pkl')

# List of all expected feature columns (as per the training)
required_columns = [
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 
    'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'mfcc_14', 'mfcc_15', 'mfcc_16', 
    'mfcc_17', 'mfcc_18', 'mfcc_19', 'mfcc_20', 'mfcc_21', 'mfcc_22', 'mfcc_23', 'mfcc_24', 
    'mfcc_25', 'mfcc_26', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'rms', 
    'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 
    'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'spectral_contrast_1', 
    'spectral_contrast_2', 'spectral_contrast_3', 'spectral_contrast_4', 'spectral_contrast_5', 
    'spectral_contrast_6', 'spectral_contrast_7', 'tonnetz_1', 'tonnetz_2', 'tonnetz_3', 
    'tonnetz_4', 'tonnetz_5', 'tonnetz_6'
]

# Function to extract features for a 7-second segment
def extract_features_segment(y, sr=16000):
    # Extract features (same as before)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=31)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # Create a feature dictionary (matching the required column names)
    features = {
        'mfcc_1': np.mean(mfccs[0]),
        'mfcc_2': np.mean(mfccs[1]),
        'mfcc_3': np.mean(mfccs[2]),
        'mfcc_4': np.mean(mfccs[3]),
        'mfcc_5': np.mean(mfccs[4]),
        'mfcc_6': np.mean(mfccs[5]),
        'mfcc_7': np.mean(mfccs[6]),
        'mfcc_8': np.mean(mfccs[7]),
        'mfcc_9': np.mean(mfccs[8]),
        'mfcc_10': np.mean(mfccs[9]),
        'mfcc_11': np.mean(mfccs[10]),
        'mfcc_12': np.mean(mfccs[11]),
        'mfcc_13': np.mean(mfccs[12]),
        'mfcc_14': np.mean(mfccs[13]),
        'mfcc_15': np.mean(mfccs[14]),
        'mfcc_16': np.mean(mfccs[15]),
        'mfcc_17': np.mean(mfccs[16]),
        'mfcc_18': np.mean(mfccs[17]),
        'mfcc_19': np.mean(mfccs[18]),
        'mfcc_20': np.mean(mfccs[19]),
        'mfcc_21': np.mean(mfccs[20]),
        'mfcc_22': np.mean(mfccs[21]),
        'mfcc_23': np.mean(mfccs[22]),
        'mfcc_24': np.mean(mfccs[23]),
        'mfcc_25': np.mean(mfccs[24]),
        'mfcc_26': np.mean(mfccs[25]),
        'zcr': np.mean(zcr),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'rms': np.mean(rms),
    }
    
    # Add chroma features
    for i in range(12):
        features[f'chroma_{i+1}'] = np.mean(chroma[i])
    
    # Add spectral contrast features
    for i in range(7):
        features[f'spectral_contrast_{i+1}'] = np.mean(spectral_contrast[i])
    
    # Add tonnetz features
    for i in range(6):
        features[f'tonnetz_{i+1}'] = np.mean(tonnetz[i])
    
    return features

# Function to reorder the extracted features to match training columns
def reorder_features(features):
    return [features.get(col, 0) for col in required_columns]

# Function to process audio and predict
# Function to process audio and predict
def process_and_predict(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=16000)
        features = extract_features_segment(y, sr)
        df = pd.DataFrame([features])

        # Preprocess features
        scaled = scaler.transform(df)
        pca_transformed = pca_model.transform(scaled)

        # Predict with all models
        predictions = {
            "RandomForest": label_encoder.inverse_transform(rf_model.predict(pca_transformed)).tolist(),
            "KNN": label_encoder.inverse_transform(knn_model.predict(pca_transformed)).tolist(),
            "XGBoost": label_encoder.inverse_transform(xgb_model.predict(pca_transformed)).tolist(),
        }
        return predictions
    except Exception as e:
        return {"error": str(e)}
