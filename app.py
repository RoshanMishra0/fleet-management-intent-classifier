from fastapi import FastAPI, HTTPException
from schemas import QueryRequest, ClassificationResponse
from llm_extractor import extract_with_llm
from postprocess import pre_classify, postprocess

app = FastAPI(
    title="Fleet Management AI Intent Classifier",
    description="Local LLM-based intent classification and slot extraction POC",
    version="1.1.0"
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fleet AI POC is running"}

@app.post("/classify", response_model=ClassificationResponse)
def classify_query(request: QueryRequest):
    try:
        # 1. Requests that fixed rules can answer never reach the model.
        ruled = pre_classify(request.query)
        if ruled is not None:
            return ClassificationResponse(**ruled)

        # 2. The model interprets the query; code enforces the business rules.
        raw = extract_with_llm(request.query)
        return ClassificationResponse(**postprocess(request.query, raw))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
