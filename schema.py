from pydantic import BaseModel, Field

class VoiceFeatures(BaseModel):
    meanfreq: float = Field(..., description="Mean frequency (in kHz)")
    sd: float = Field(..., description="Standard deviation of frequency")
    median: float = Field(..., description="Median frequency (in kHz)")
    Q25: float = Field(..., description="First quantile (in kHz)")
    Q75: float = Field(..., description="Third quantile (in kHz)")
    IQR: float = Field(..., description="Interquartile range (in kHz)")
    skew: float = Field(..., description="Skewness")
    kurt: float = Field(..., description="Kurtosis")
    sp_ent: float = Field(..., description="Spectral entropy")
    sfm: float = Field(..., description="Spectral flatness")
    mode: float = Field(..., description="Mode frequency")
    centroid: float = Field(..., description="Frequency centroid")
    meanfun: float = Field(..., description="Mean fundamental frequency measured across acoustic signal")
    minfun: float = Field(..., description="Minimum fundamental frequency measured across acoustic signal")
    maxfun: float = Field(..., description="Maximum fundamental frequency measured across acoustic signal")
    meandom: float = Field(..., description="Mean of dominant frequency measured across acoustic signal")
    mindom: float = Field(..., description="Minimum of dominant frequency measured across acoustic signal")
    maxdom: float = Field(..., description="Maximum of dominant frequency measured across acoustic signal")
    dfrange: float = Field(..., description="Range of dominant frequency measured across acoustic signal")
    modindx: float = Field(..., description="Modulation index")

class GenderPrediction(BaseModel):
    probability: float
    gender: str
    uncertainty_raw: float
    uncertainty_percent: float
    confidence: float