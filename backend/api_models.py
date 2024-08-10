from pydantic import BaseModel


class PredictionRequest(BaseModel):
    trip_id: str
    request_datetime: str
    trip_distance: float
    PULocationID: int
    DOLocationID: int
    Airport: int


class PredictionResponse(BaseModel):
    prediction: float
