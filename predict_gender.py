from http.client import HTTPException
from typing import Dict, List, Optional, Any
from venv import logger
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import traceback


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



def extract_features_from_audio(audio_file):
    """Extract voice features from audio file using librosa"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract features similar to the training data
        # Frequency features
        meanfreq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sd = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        median = np.median(spec_cent)
        Q25 = np.percentile(spec_cent, 25)
        Q75 = np.percentile(spec_cent, 75)
        IQR = Q75 - Q25
        
        # Spectral statistics
        spec = np.abs(librosa.stft(y))
        skew = np.mean(librosa.feature.spectral_contrast(S=spec, sr=sr))
        kurt = np.std(librosa.feature.spectral_contrast(S=spec, sr=sr))
        
        # Spectral entropy and flatness
        spec_entropy = np.mean(librosa.feature.spectral_flatness(y=y))
        sfm = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Mode and centroid
        mode = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Fundamental frequency statistics
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches) > 0:
            meanfun = np.mean(pitches)
            minfun = np.min(pitches)
            maxfun = np.max(pitches)
        else:
            meanfun = 0
            minfun = 0
            maxfun = 0
        
        # Dominant frequency statistics
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        dom_freqs = []
        for i in range(S.shape[1]):
            if np.sum(S[:, i]) > 0:  # Avoid empty frames
                dom_freq = freqs[np.argmax(S[:, i])]
                dom_freqs.append(dom_freq)
        
        if len(dom_freqs) > 0:
            meandom = np.mean(dom_freqs)
            mindom = np.min(dom_freqs)
            maxdom = np.max(dom_freqs)
            dfrange = maxdom - mindom
        else:
            meandom = 0
            mindom = 0
            maxdom = 0
            dfrange = 0
        
        # Modulation index
        modindx = np.mean(librosa.feature.rms(y=y))
        
        # Create feature dictionary
        features = {
            'meanfreq': float(meanfreq),
            'sd': float(sd),
            'median': float(median),
            'Q25': float(Q25),
            'Q75': float(Q75),
            'IQR': float(IQR),
            'skew': float(skew),
            'kurt': float(kurt),
            'sp_ent': float(spec_entropy),
            'sfm': float(sfm),
            'mode': float(mode),
            'centroid': float(centroid),
            'meanfun': float(meanfun),
            'minfun': float(minfun),
            'maxfun': float(maxfun),
            'meandom': float(meandom),
            'mindom': float(mindom),
            'maxdom': float(maxdom),
            'dfrange': float(dfrange),
            'modindx': float(modindx)
        }
        
        # Normalize if feature means and stds are available
        if feature_means is not None and feature_stds is not None:
            features_array = np.array([list(features.values())])
            normalized_features = (features_array - feature_means) / feature_stds
            
            # Convert back to dictionary
            normalized_dict = {k: float(v) for k, v in zip(features.keys(), normalized_features[0])}
            return normalized_dict
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error extracting audio features: {str(e)}")
