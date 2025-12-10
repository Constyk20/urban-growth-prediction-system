# main.py — Urban Growth Prediction API (FastAPI)
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import shutil
import json
from motor.motor_asyncio import AsyncIOMotorClient
from redis import Redis
from rq import Queue
import asyncio

app = FastAPI(title="Urban Growth Prediction System", version="1.0")

# CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB
client = AsyncIOMotorClient(os.getenv("MONGO_URI", "mongodb+srv://lwlblessings_db_user:AaKJxe3IH6Gj5VmO@urban-growth.cshbjfi.mongodb.net/?appName=urban-growth"))
db = client.urban_growth
tasks = db.tasks

# Redis + RQ Queue
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://default:AUuWAAIncDI0NmNhNGExMDFkMDg0Njg0YTA0OTZlNWNiMWQ0ODVlY3AyMTkzNTA@artistic-boxer-19350.upstash.io:6379"))
q = Queue("urban_predict", connection=redis_conn)

class AOIRequest(BaseModel):
    aoi: dict

# Routes
@app.get("/")
async def root():
    return {"message": "Urban Growth Prediction API - LIVE"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip files allowed")
    
    task_id = str(uuid.uuid4())
    os.makedirs("uploads/raw-imagery", exist_ok=True)
    file_path = f"uploads/raw-imagery/{task_id}_{file.filename}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    await tasks.insert_one({
        "task_id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "file_path": file_path,
        "result": None
    })
    
    return {"task_id": task_id, "message": "File uploaded"}

@app.post("/api/predict/{task_id}")
async def predict(task_id: str, req: AOIRequest):
    task = await tasks.find_one({"task_id": task_id})
    if not task or task["status"] != "uploaded":
        raise HTTPException(400, "Task not found or already processed")
    
    # Enqueue prediction job
    job = q.enqueue("ml.predict.run_prediction", task["file_path"], req.aoi)
    
    await tasks.update_one(
        {"task_id": task_id},
        {"$set": {"status": "queued", "rq_job_id": job.id}}
    )
    
    return {"task_id": task_id, "job_id": job.id, "status": "queued"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = await tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(404, "Task not found")
    
    if task["status"] == "completed":
        return task
    
    # Check RQ job status
    if "rq_job_id" in task:
        from rq.job import Job
        try:
            job = Job.fetch(task["rq_job_id"], connection=redis_conn)
            if job.is_finished:
                result = job.result
                await tasks.update_one(
                    {"task_id": task_id},
                    {"$set": {"status": "completed", "result": result}}
                )
                task["result"] = result
                task["status"] = "completed"
        except:
            pass
    
    return task