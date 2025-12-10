# main.py — 100% MOCK NIGERIA-BASED (no Redis, no DB, no real ML)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import asyncio

app = FastAPI(title="Urban Growth Predictor - Nigeria")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nigerian cities with mock growth data
NIGERIA_CITIES = {
    "Lagos": {"current": 12.8, "predicted": 18.5, "growth": 45},
    "Abuja": {"current": 4.2, "predicted": 7.1, "growth": 69},
    "Kano": {"current": 3.9, "predicted": 5.8, "growth": 49},
    "Ibadan": {"current": 3.5, "predicted": 5.2, "growth": 49},
    "Port Harcourt": {"current": 2.8, "predicted": 4.6, "growth": 64},
    "Enugu": {"current": 1.1, "predicted": 2.3, "growth": 109},
    "Kaduna": {"current": 1.6, "predicted": 2.9, "growth": 81},
}

class AOIRequest(BaseModel):
    city: str = "Lagos"
    year: int = 2030

@app.get("/")
async def root():
    return {"message": "Urban Growth Predictor - Nigeria (Mock Demo)"}

@app.post("/api/predict")
async def predict(req: AOIRequest):
    city = req.city.title()
    if city not in NIGERIA_CITIES:
        raise HTTPException(400, f"City '{city}' not supported")
    
    data = NIGERIA_CITIES[city]
    
    # Simulate processing time
    await asyncio.sleep(3)
    
    return {
        "city": city,
        "currentBuiltUpAreaHa": data["current"] * 1000,
        "predictedBuiltUpAreaHa": data["predicted"] * 1000,
        "growthPercent": data["growth"],
        "confidence": round(random.uniform(0.87, 0.96), 2),
        "year": req.year,
        "resultUrl": "https://via.placeholder.com/800x600/228B22/FFFFFF?text=Urban+Growth+" + city.replace(" ", "+"),
        "message": f"Rapid urbanization predicted in {city}!"
    }

@app.get("/api/cities")
async def get_cities():
    return {"cities": list(NIGERIA_CITIES.keys())}