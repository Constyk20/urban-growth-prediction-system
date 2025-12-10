# main.py — AI-Enhanced Urban Growth Predictor for Nigeria
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import random
import asyncio
from datetime import datetime
import math

app = FastAPI(
    title="AI Urban Growth Predictor",
    description="Advanced ML-powered urban expansion forecasting for Nigerian cities",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Nigerian cities with richer mock AI data
NIGERIA_CITIES = {
    "Lagos": {
        "current": 12.8,
        "predicted": 18.5,
        "growth": 45,
        "population": 15_300_000,
        "density": 20000,
        "lat": 6.5244,
        "lng": 3.3792,
        "features": ["coastal", "economic_hub", "high_traffic"]
    },
    "Abuja": {
        "current": 4.2,
        "predicted": 7.1,
        "growth": 69,
        "population": 3_800_000,
        "density": 9500,
        "lat": 9.0765,
        "lng": 7.3986,
        "features": ["capital", "planned_city", "rapid_development"]
    },
    "Kano": {
        "current": 3.9,
        "predicted": 5.8,
        "growth": 49,
        "population": 4_100_000,
        "density": 15000,
        "lat": 12.0022,
        "lng": 8.5920,
        "features": ["historic", "commercial", "northern_hub"]
    },
    "Ibadan": {
        "current": 3.5,
        "predicted": 5.2,
        "growth": 49,
        "population": 3_600_000,
        "density": 12000,
        "lat": 7.3775,
        "lng": 3.9470,
        "features": ["historic", "educational", "sprawling"]
    },
    "Port Harcourt": {
        "current": 2.8,
        "predicted": 4.6,
        "growth": 64,
        "population": 3_200_000,
        "density": 14000,
        "lat": 4.8156,
        "lng": 7.0498,
        "features": ["oil_hub", "industrial", "coastal"]
    },
    "Enugu": {
        "current": 1.1,
        "predicted": 2.3,
        "growth": 109,
        "population": 820_000,
        "density": 8500,
        "lat": 6.5244,
        "lng": 7.5105,
        "features": ["coal_city", "emerging", "hilly_terrain"]
    },
    "Kaduna": {
        "current": 1.6,
        "predicted": 2.9,
        "growth": 81,
        "population": 1_900_000,
        "density": 11000,
        "lat": 10.5105,
        "lng": 7.4165,
        "features": ["industrial", "military", "textile_hub"]
    },
}

# AI Model simulation states
class AIModelState:
    def __init__(self):
        self.training_progress = 0
        self.is_training = False
        self.accuracy = 0.94
        self.model_version = "v2.3.1"
        self.last_updated = datetime.now()

ai_model = AIModelState()

class PredictionRequest(BaseModel):
    city: str = Field(..., description="Nigerian city name")
    year: int = Field(2030, ge=2025, le=2050, description="Target prediction year")
    scenario: Optional[str] = Field("moderate", description="Growth scenario: conservative, moderate, aggressive")
    include_analytics: bool = Field(True, description="Include detailed AI analytics")

class GrowthDriver(BaseModel):
    factor: str
    impact: float
    confidence: float
    description: str

class PredictionResponse(BaseModel):
    city: str
    year: int
    scenario: str
    currentBuiltUpAreaKm2: float
    predictedBuiltUpAreaKm2: float
    growthPercent: float
    confidence: float
    modelVersion: str
    timestamp: str
    
    # AI Analytics
    growthDrivers: List[GrowthDriver]
    riskFactors: List[str]
    opportunities: List[str]
    
    # Spatial data
    expansionZones: List[Dict]
    populationDensityChange: float
    infrastructureScore: float
    
    # Visualization
    resultUrl: str
    heatmapUrl: str
    timelapseUrl: str

def simulate_ai_processing(city: str, year: int, scenario: str) -> Dict:
    """Simulate complex AI model inference with realistic processing"""
    
    base_data = NIGERIA_CITIES[city]
    years_ahead = year - 2024
    
    # Scenario multipliers
    multipliers = {
        "conservative": 0.7,
        "moderate": 1.0,
        "aggressive": 1.4
    }
    mult = multipliers.get(scenario, 1.0)
    
    # Calculate growth with AI-like non-linear scaling
    base_growth = base_data["growth"]
    adjusted_growth = base_growth * mult * (1 + math.log(years_ahead) / 10)
    
    predicted_area = base_data["current"] * (1 + adjusted_growth / 100)
    
    # Generate AI-driven insights
    growth_drivers = [
        GrowthDriver(
            factor="Population Migration",
            impact=round(random.uniform(0.3, 0.5), 2),
            confidence=round(random.uniform(0.85, 0.95), 2),
            description=f"Rural-urban migration driving {random.randint(25, 40)}% of expansion"
        ),
        GrowthDriver(
            factor="Economic Development",
            impact=round(random.uniform(0.2, 0.4), 2),
            confidence=round(random.uniform(0.80, 0.92), 2),
            description=f"GDP growth correlation: {random.uniform(0.65, 0.85):.2f}"
        ),
        GrowthDriver(
            factor="Infrastructure Projects",
            impact=round(random.uniform(0.15, 0.35), 2),
            confidence=round(random.uniform(0.75, 0.90), 2),
            description=f"Planned transport corridors influencing {random.randint(15, 30)}% of growth"
        ),
    ]
    
    risk_factors = [
        "Flood-prone areas in expansion zones",
        "Infrastructure capacity constraints",
        "Informal settlement growth",
        "Environmental degradation risks"
    ]
    
    opportunities = [
        "Planned urban development corridors",
        "Smart city integration potential",
        "Green infrastructure opportunities",
        "Mixed-use development zones"
    ]
    
    # Generate expansion zones (mock AI spatial analysis)
    expansion_zones = [
        {
            "zone": f"{city} {direction}",
            "probability": round(random.uniform(0.6, 0.95), 2),
            "areaKm2": round(random.uniform(5, 25), 1),
            "type": random.choice(["residential", "commercial", "industrial", "mixed"])
        }
        for direction in ["North", "South", "East", "West"]
    ]
    
    return {
        "predicted_area": predicted_area,
        "adjusted_growth": adjusted_growth,
        "growth_drivers": growth_drivers,
        "risk_factors": risk_factors[:3],
        "opportunities": opportunities[:3],
        "expansion_zones": expansion_zones,
        "density_change": round(random.uniform(-15, 25), 1),
        "infrastructure_score": round(random.uniform(0.55, 0.85), 2)
    }

@app.get("/")
async def root():
    return {
        "service": "AI Urban Growth Predictor",
        "region": "Nigeria",
        "model": ai_model.model_version,
        "accuracy": ai_model.accuracy,
        "status": "online",
        "capabilities": [
            "Multi-scenario forecasting",
            "Spatial growth analysis",
            "Risk assessment",
            "Infrastructure impact modeling"
        ]
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, background_tasks: BackgroundTasks):
    city = req.city.title()
    
    if city not in NIGERIA_CITIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"City '{city}' not in our model",
                "available_cities": list(NIGERIA_CITIES.keys()),
                "suggestion": "Try one of the available cities"
            }
        )
    
    # Simulate AI model inference time
    await asyncio.sleep(random.uniform(2.5, 4.5))
    
    # Run AI simulation
    ai_results = simulate_ai_processing(city, req.year, req.scenario)
    
    base_data = NIGERIA_CITIES[city]
    
    response = PredictionResponse(
        city=city,
        year=req.year,
        scenario=req.scenario,
        currentBuiltUpAreaKm2=base_data["current"],
        predictedBuiltUpAreaKm2=round(ai_results["predicted_area"], 2),
        growthPercent=round(ai_results["adjusted_growth"], 1),
        confidence=round(random.uniform(0.87, 0.96), 2),
        modelVersion=ai_model.model_version,
        timestamp=datetime.now().isoformat(),
        
        growthDrivers=ai_results["growth_drivers"],
        riskFactors=ai_results["risk_factors"],
        opportunities=ai_results["opportunities"],
        
        expansionZones=ai_results["expansion_zones"],
        populationDensityChange=ai_results["density_change"],
        infrastructureScore=ai_results["infrastructure_score"],
        
        resultUrl=f"https://via.placeholder.com/1200x800/228B22/FFFFFF?text={city.replace(' ', '+')}+Growth+Map+{req.year}",
        heatmapUrl=f"https://via.placeholder.com/1200x800/FF6347/FFFFFF?text={city.replace(' ', '+')}+Density+Heatmap",
        timelapseUrl=f"https://via.placeholder.com/1200x800/4169E1/FFFFFF?text={city.replace(' ', '+')}+Growth+Timelapse"
    )
    
    # Log prediction in background
    background_tasks.add_task(log_prediction, city, req.year, req.scenario)
    
    return response

@app.get("/api/cities")
async def get_cities():
    return {
        "cities": [
            {
                "name": name,
                "population": data["population"],
                "currentAreaKm2": data["current"],
                "features": data["features"],
                "coordinates": {"lat": data["lat"], "lng": data["lng"]}
            }
            for name, data in NIGERIA_CITIES.items()
        ],
        "total": len(NIGERIA_CITIES)
    }

@app.get("/api/model/status")
async def model_status():
    return {
        "version": ai_model.model_version,
        "accuracy": ai_model.accuracy,
        "training": ai_model.is_training,
        "lastUpdated": ai_model.last_updated.isoformat(),
        "metrics": {
            "mae": round(random.uniform(0.05, 0.12), 3),
            "rmse": round(random.uniform(0.08, 0.15), 3),
            "r2_score": round(random.uniform(0.88, 0.94), 3)
        },
        "trainingData": {
            "cities": len(NIGERIA_CITIES),
            "timeRange": "2000-2024",
            "samples": random.randint(5000, 8000)
        }
    }

@app.post("/api/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    if ai_model.is_training:
        raise HTTPException(400, "Model is already training")
    
    background_tasks.add_task(simulate_training)
    
    return {
        "status": "training_started",
        "message": "AI model retraining initiated",
        "estimatedTime": "15-20 minutes"
    }

@app.get("/api/compare")
async def compare_cities(cities: str, year: int = 2030):
    """Compare growth predictions across multiple cities"""
    city_list = [c.strip().title() for c in cities.split(",")]
    
    invalid_cities = [c for c in city_list if c not in NIGERIA_CITIES]
    if invalid_cities:
        raise HTTPException(400, f"Invalid cities: {', '.join(invalid_cities)}")
    
    await asyncio.sleep(2)
    
    comparisons = []
    for city in city_list:
        data = NIGERIA_CITIES[city]
        years_ahead = year - 2024
        growth = data["growth"] * (1 + math.log(years_ahead) / 10)
        
        comparisons.append({
            "city": city,
            "currentKm2": data["current"],
            "predictedKm2": round(data["current"] * (1 + growth / 100), 2),
            "growthRate": round(growth, 1),
            "rank": 0  # Will be assigned after sorting
        })
    
    # Rank by growth rate
    comparisons.sort(key=lambda x: x["growthRate"], reverse=True)
    for idx, comp in enumerate(comparisons):
        comp["rank"] = idx + 1
    
    return {
        "year": year,
        "cities": comparisons,
        "analysis": f"Based on AI model {ai_model.model_version}",
        "timestamp": datetime.now().isoformat()
    }

async def log_prediction(city: str, year: int, scenario: str):
    """Background task to log predictions"""
    await asyncio.sleep(1)
    print(f"[LOG] Prediction: {city}, Year: {year}, Scenario: {scenario}")

async def simulate_training():
    """Simulate AI model training process"""
    ai_model.is_training = True
    ai_model.training_progress = 0
    
    for i in range(10):
        await asyncio.sleep(2)
        ai_model.training_progress = (i + 1) * 10
        print(f"[TRAINING] Progress: {ai_model.training_progress}%")
    
    ai_model.accuracy = round(random.uniform(0.94, 0.97), 3)
    ai_model.last_updated = datetime.now()
    ai_model.is_training = False
    print("[TRAINING] Complete!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)