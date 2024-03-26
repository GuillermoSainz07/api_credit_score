from fastapi import APIRouter
from schemas.features import Features
from config_db.database import Session
from services.prediction import ModelPrediction
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder


pred_route = APIRouter()

@pred_route.post("/predict_model", tags=['Prediction'], response_model=dict)
def make_prediction(features:Features) -> dict:
    db = Session()
    resul = ModelPrediction(db).make_prediction(**features.model_dump())
    predict_label = {0:'Poor',
                         1:'Standard',
                         2:'Good'}
    result_str = predict_label[resul]

    return JSONResponse(content={'Prediction': str(result_str)})