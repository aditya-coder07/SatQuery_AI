from fastapi import FastAPI

app = FastAPI(title="SatQuery AI API")

@app.get("/health")
def health_check():
    return {"status": "ok"}
