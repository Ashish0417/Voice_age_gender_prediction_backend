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



