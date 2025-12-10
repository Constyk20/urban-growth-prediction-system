# main.py — Urban Growth Prediction API (FastAPI + RQ Worker in ONE Process)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import shutil
import json
import threading
from motor.motor_asyncio import AsyncIOMotorClient
from redis import Redis
from rq import Queue
from rq.job import Job

# === FastAPI App ===
app = FastAPI(title="Urban Growth Prediction System", version="1.0")

# CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Database & Queue ===
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://lwlblessings_db_user:AaKJxe3IH6Gj5VmO@urban-growth.cshbjfi.mongodb.net/?appName=urban-growth")
REDIS_URL = os.getenv("REDIS_URL", "redis://default:AUuWAAIncDI0NmNhNGExMDFkMDg0Njg0YTA0OTZlNWNiMWQ0ODVlY3AyMTkzNTA@artistic-boxer-19350.upstash.io:6379")

client = AsyncIOMotorClient(MONGO_URI)
db = client.urban_growth
tasks = db.tasks

redis_conn = Redis.from_url(REDIS_URL)
q = Queue("urban_predict", connection=redis_conn)

# === Import ML Function ===
from ml.predict import run_prediction

# === Request Model ===
class AOIRequest(BaseModel):
    aoi: dict

# === Routes ===
@app.get("/")
async def root():
    return {"message": "Urban Growth API LIVE — Ready for Flutter"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
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
    
    return {"task_id": task_id, "message": "File uploaded successfully"}

@app.post("/api/predict/{task_id}")
async def predict(task_id: str, req: AOIRequest):
    task = await tasks.find_one({"task_id": task_id})
    if not task or task["status"] != "uploaded":
        raise HTTPException(400, "Invalid task or already processed")
    
    # Enqueue job
    job = q.enqueue(run_prediction, task["file_path"], json.dumps(req.aoi))
    
    await tasks.update_one(
        {"task_id": task_id},
        {"$set": {"status": "processing", "job_id": job.id}}
    )
    
    return {"task_id": task_id, "job_id": job.id, "status": "processing"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = await tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(404, "Task not found")
    
    # Check job status
    if task["status"] == "processing" and "job_id" in task:
        try:
            job = Job.fetch(task["job_id"], connection=redis_conn)
            if job.is_finished:
                result = job.result
                await tasks.update_one(
                    {"task_id": task_id},
                    {"$set": {"status": "completed", "result": result}}
                )
                task["status"] = "completed"
                task["result"] = result
        except:
            pass
    
    return task

# === BACKGROUND RQ WORKER (Runs in same process) ===
def start_worker():
    from rq.worker import SimpleWorker
    worker = SimpleWorker([q], connection=redis_conn)
    print("RQ Worker STARTED — Processing jobs...")
    worker.work()

# Start worker in background thread
threading.Thread(target=start_worker, daemon=True).start()

print("FastAPI + RQ Worker READY")