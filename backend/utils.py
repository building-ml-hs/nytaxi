import json
import logging
import pandas as pd
import numpy as np


def log_features_and_prediction(
    extracted_features: pd.DataFrame, prediction: float, trip_id: str
) -> None:
    logger = logging.getLogger(__name__)

    flattened_data = {}
    for key, value in extracted_features.items():
        if isinstance(value, np.ndarray):
            flattened_data[key] = value[0]
        elif isinstance(value, pd.Series):
            flattened_data[key] = value.iloc[0]
        else:
            flattened_data[key] = value

    result = {
        "extracted_features": flattened_data,
        "prediction": float(prediction),
        "trip_id": trip_id,
    }
    result_json_str = json.dumps(result, separators=(",", ":"))

    logger.info(f"prediction_result: {result_json_str}")
