# main.py - FastAPI implementation for Voice Gender Classification

import sys
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import librosa
import tempfile
import os
import json
import traceback
import shutil
import logging

from schema import GenderPrediction, VoiceFeatures
from predict_gender import engineer_features,create_ann_model, extract_features_from_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow and other warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING messages

# Initialize FastAPI app
app = FastAPI(
    title="Voice Analysis API",
    description="API for gender prediction from voice features with uncertainty estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define input and output models

    
def predict_with_bagging_ensemble(models, X, n_simulations=30, noise_scale=0.08):
    """
    Predict using the bagging ensemble with noise-based uncertainty estimation
    """
    predictions = []
    
    for _ in range(n_simulations):
        # Add random Gaussian noise to the features
        X_noisy = X + np.random.normal(0, noise_scale, X.shape)
        
        # Get predictions from all models
        model_preds = []
        for name, model in models.items():
            if name == 'ann':
                # For Keras models wrapped in BaggingClassifier (if available)
                try:
                    pred = model.predict_proba(X_noisy)[:, 1]
                except AttributeError:
                    # If model doesn't have predict_proba, use predict
                    pred = model.predict(X_noisy)
            else:
                # For other sklearn models
                try:
                    pred = model.predict_proba(X_noisy)[:, 1]
                except AttributeError:
                    # If model doesn't have predict_proba, use predict
                    pred = model.predict(X_noisy)
            model_preds.append(pred)
        
        # Average predictions across models
        avg_pred = np.mean(model_preds, axis=0)
        predictions.append(avg_pred)
    
    # Convert to numpy array
    predictions = np.array(predictions)
    
    # Calculate mean and standard deviation
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Normalize uncertainty to percentage
    uncertainty_percent = (std_pred / max_uncertainty) * 100
    uncertainty_percent = np.minimum(uncertainty_percent, 100)  # Cap at 100%
    
    return mean_pred, std_pred, uncertainty_percent


# Load all models at startup
@app.on_event("startup")
async def load_models():
    global gender_scaler, gender_label_encoder, bagging_models
    global feature_means, feature_stds, max_uncertainty
    
    try:
        # Load gender classification models - using your existing files
        gender_scaler = joblib.load("scaler.pkl")
        gender_label_encoder = joblib.load("label_encoder.pkl")
        sys.modules['__main__'].create_ann_model = create_ann_model
        
        # Load bagging models
        bagging_models = joblib.load("bagging_models.pkl")
        logger.info("Bagging models loaded successfully")
        
        # Load feature normalization values
        feature_means = np.load("feature_means.npy")
        feature_stds = np.load("feature_stds.npy")
        logger.info("Feature normalization values loaded successfully")
            
        # Set default max uncertainty value for normalization
        max_uncertainty = 1.0
        logger.info("Using default max uncertainty value")
            
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        raise




# Helper functions

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

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Voice Gender Analysis API is running. Access /docs for API documentation."}

@app.post("/predict/gender", response_model=GenderPrediction)
async def predict_gender(features: VoiceFeatures):
    """
    Predict gender from voice features
    """
    try:
        # Convert Pydantic model to dictionary
        features_dict = features.dict()
        
        # Extract and prepare features for prediction
        feature_names = list(features_dict.keys())
        features_list = [features_dict[name] for name in feature_names]
        X = np.array(features_list).reshape(1, -1)
        
        # Scale features
        X_scaled = gender_scaler.transform(X)
        
        # Add engineered features
        X_eng = engineer_features(X_scaled)
        
        # Make prediction using bagging ensemble
        pred_mean, pred_std, uncertainty_percent = predict_with_bagging_ensemble(
            bagging_models, X_eng
        )
        
        # Determine gender
        gender = "male" if pred_mean[0] > 0.5 else "female"
        
        # Calculate confidence
        confidence = 100 - float(uncertainty_percent[0])
        
        return {
            "probability": float(pred_mean[0]),
            "gender": gender,
            "uncertainty_raw": float(pred_std[0]),
            "uncertainty_percent": float(uncertainty_percent[0]),
            "confidence": confidence  # <-- Fixed: removed [0]
        }
    except Exception as e:
        logger.error(f"Error predicting gender: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error predicting gender: {str(e)}")

@app.post("/analyze/audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """
    Extract features from audio file and predict gender
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
            # Write content of uploaded file to temp file
            shutil.copyfileobj(audio_file.file, tmp)
            temp_path = tmp.name
        
        try:
            # Extract features from audio file
            features = extract_features_from_audio(temp_path)
            
            # Convert dict to Pydantic model
            features_model = VoiceFeatures(**features)
            
            # Make predictions
            result = await predict_gender(features_model)
            
            # Add extracted features to response
            response = {
                "gender_prediction": result,
                "extracted_features": features
            }
            
            return response
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)