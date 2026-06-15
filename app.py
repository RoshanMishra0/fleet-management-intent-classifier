from fastapi import FastAPI, HTTPException
from schemas import QueryRequest, ClassificationResponse
from llm_extractor import extract_with_llm

app = FastAPI(
    title="Fleet Management AI Intent Classifier",
    description="Local LLM-based intent classification and slot extraction POC",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fleet AI POC is running"}

@app.post("/classify", response_model=ClassificationResponse)
def classify_query(request: QueryRequest):
    try:
        result = extract_with_llm(request.query)
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))