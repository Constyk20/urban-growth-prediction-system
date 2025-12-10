web: uvicorn main:app --host 0.0.0.0 --port $PORT
worker: python -c "from main import start_worker; start_worker()"
release: echo "Release phase (optional)"