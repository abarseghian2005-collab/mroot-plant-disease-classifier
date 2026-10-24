"""FastAPI service for local or container deployment."""

from functools import lru_cache
import os
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .config import load_config
from .validation import InvalidImage, MAX_IMAGE_BYTES, decode_image


class DiseaseScore(BaseModel):
    disease: str
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: str
    top5: list[DiseaseScore] = Field(min_length=5, max_length=5)
    model_version: str = "2.0.0"


def create_app(predictor_factory=None):
    application = FastAPI(title="MROOT plant disease classifier", version="2.0.0")
    inference_lock = threading.Lock()

    @lru_cache(maxsize=1)
    def predictor():
        if predictor_factory is not None:
            return predictor_factory()
        from .assets import ensure_model_assets
        from .inference import MrootEnsemble
        return MrootEnsemble(ensure_model_assets(), device=os.environ.get("MROOT_DEVICE", "cpu"))

    @application.get("/health")
    def health():
        return {"status": "ok", "classes": len(load_config()["classes"]), "model_version": "2.0.0"}

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        data = await file.read(MAX_IMAGE_BYTES + 1)
        try:
            image = decode_image(data)
        except InvalidImage as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        def classify():
            with inference_lock:
                scores = predictor().predict(image)
            top5 = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
            return PredictionResponse(
                prediction=top5[0][0],
                top5=[DiseaseScore(disease=name, probability=value) for name, value in top5],
            )

        return await run_in_threadpool(classify)

    return application


app = create_app()
