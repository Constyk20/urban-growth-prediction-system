# ml/worker.py — RQ worker (runs every minute via Render Cron Job)
from rq import Queue, Connection
from redis import Redis
import os
from ml.predict import run_prediction

redis_url = os.getenv("REDIS_URL", "redis://default:AUuWAAIncDI0NmNhNGExMDFkMDg0Njg0YTA0OTZlNWNiMWQ0ODVlY3AyMTkzNTA@artistic-boxer-19350.upstash.io:6379")
q = Queue("urban_predict", connection=Redis.from_url(redis_url))

def main():
    print("ML Worker running — checking queue...")
    job = q.dequeue()
    if job:
        print(f"Processing job {job.id}")
        result = run_prediction(job.kwargs["zip_url"], job.kwargs["aoi_json"])
        job.save()
        job.result = result
        job.status = "finished"

if __name__ == "__main__":
    main()