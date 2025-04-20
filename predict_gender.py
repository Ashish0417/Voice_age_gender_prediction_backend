from http.client import HTTPException
from typing import Dict, List, Optional, Any
from venv import logger
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import traceback
import librosa
import numpy as np
import tempfile
from fastapi import UploadFile
import soundfile as sf
import scipy.stats
from fastapi import HTTPException
import numpy as np
import joblib
import tensorflow as tf
import os
import pandas as pd

async def extract_voice_features_from_mp3(file: UploadFile) -> dict:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Load audio using librosa
    y, sr = librosa.load(temp_file_path, sr=None)

    # Ensure audio is long enough
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), mode='constant')

    # Extract features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    sfm = librosa.feature.spectral_flatness(y=y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # Fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) == 0:
        pitches = np.array([0])

    # Dominant frequency
    freqs = np.abs(np.fft.rfft(y))
    dom_freqs = np.fft.rfftfreq(len(y), d=1/sr)

    # Features
    meanfreq = np.mean(dom_freqs)
    sd = np.std(dom_freqs)
    median = np.median(dom_freqs)
    Q25 = np.percentile(dom_freqs, 25)
    Q75 = np.percentile(dom_freqs, 75)
    IQR = Q75 - Q25
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    sp_ent = scipy.stats.entropy(np.abs(freqs))
    sfm_val = np.mean(sfm)
    mode = dom_freqs[np.argmax(freqs)]
    centroid_val = np.mean(centroid)
    meanfun = np.mean(pitches)
    minfun = np.min(pitches)
    maxfun = np.max(pitches)
    meandom = np.mean(dom_freqs)
    mindom = np.min(dom_freqs)
    maxdom = np.max(dom_freqs)
    dfrange = maxdom - mindom
    modindx = np.std(pitches) / (np.mean(pitches) + 1e-6)

    features = {
        'meanfreq': float(meanfreq),
        'sd': float(sd),
        'median': float(median),
        'Q25': float(Q25),
        'Q75': float(Q75),
        'IQR': float(IQR),
        'skew': float(skew),
        'kurt': float(kurt),
        'sp.ent': float(sp_ent),
        'sfm': float(sfm_val),
        'mode': float(mode),
        'centroid': float(centroid_val),
        'meanfun': float(meanfun),
        'minfun': float(minfun),
        'maxfun': float(maxfun),
        'meandom': float(meandom),
        'mindom': float(mindom),
        'maxdom': float(maxdom),
        'dfrange': float(dfrange),
        'modindx': float(modindx)
    }

    return features

def create_ann_model(input_dim):
    """Create a neural network model for voice classification"""
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def engineer_features(X):
    """Add engineered features to improve prediction accuracy"""
    X_df = pd.DataFrame(X)
    
    # Create additional features
    try:
        # If X is a numpy array with unnamed columns
        X_df['meanfun_to_meanfreq'] = X_df[12] / X_df[0]  # meanfun to meanfreq ratio
        X_df['IQR_to_sd'] = X_df[5] / X_df[1]  # IQR to sd ratio
        X_df['maxdom_to_mindom'] = X_df[17] / X_df[16]  # maxdom to mindom ratio
        X_df['meanfreq_times_sd'] = X_df[0] * X_df[1]  # meanfreq * sd
        X_df['centroid_minus_meanfreq'] = X_df[11] - X_df[0]  # centroid - meanfreq
    except:
        # If X is a DataFrame with named columns
        try:
            X_df['meanfun_to_meanfreq'] = X_df['meanfun'] / X_df['meanfreq']
            X_df['IQR_to_sd'] = X_df['IQR'] / X_df['sd']
            X_df['maxdom_to_mindom'] = X_df['maxdom'] / X_df['mindom']
            X_df['meanfreq_times_sd'] = X_df['meanfreq'] * X_df['sd']
            X_df['centroid_minus_meanfreq'] = X_df['centroid'] - X_df['meanfreq']
        except Exception as e:
            logger.warning(f"Error creating engineered features: {str(e)}")
    
    # Handle potential infinities or NaNs
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X_df.values



