import xgboost as xgb
from typing import List


class ModelExecutor:
    def __init__(self, model_path: str):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, data: List[List[float]]) -> List[float]:
        dmatrix = xgb.DMatrix(data)
        predictions = self.model.predict(dmatrix)
        return predictions.tolist()
