from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd

app = FastAPI(title="Kaggle Housing Price Predictor API")

# 1. Poora Pipeline load karein (Isme preprocessor aur model dono hain)
pipeline = joblib.load('Housing_price_model.pkl')

# 2. Dynamic Input structure (User direct JSON bhejega)
class HouseInput(BaseModel):
    features: Dict[str, Any] # User key-value pairs mein poori row bhejega

@app.post("/predict_price")
def predict_price(data: HouseInput):
    try:
        # User ke data ko Pandas DataFrame mein badlein (Kyunki pipeline ko DataFrame chahiye)
        input_df = pd.DataFrame([data.features])
        
        # Pipeline khud scale bhi karega aur categorical data ko handle bhi karega
        prediction = pipeline.predict(input_df)
        
        return {
            "status": "success",
            "predicted_price": float(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is Live!"}
