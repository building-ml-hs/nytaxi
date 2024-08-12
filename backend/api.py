from fastapi import APIRouter, Depends
from backend.model_executor import ModelExecutor
from backend.feature_extractor import FeatureExtractor
from backend.api_models import PredictionRequest, PredictionResponse
from backend.utils import log_features_and_prediction

router = APIRouter()

_model_executor = None
_feature_extractor = None


def get_model_executor() -> ModelExecutor:
    global _model_executor

    if _model_executor is None:
        _model_executor = ModelExecutor(model_path="models/xgb.json")

    return _model_executor


def get_feature_extractor() -> FeatureExtractor:
    global _feature_extractor

    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor(
            zones_filename="zones.csv",
            data_folder="data",
            test_filename="test.csv",
        )

    return _feature_extractor


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_executor: ModelExecutor = Depends(get_model_executor),
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor),
):
    extracted_features = feature_extractor.extract_features(request)
    prediction = model_executor.predict(extracted_features)

    log_features_and_prediction(
        extracted_features=extracted_features,
        prediction=prediction,
        trip_id=request.trip_id,
    )

    return PredictionResponse(
        prediction=prediction,
    )
