import os
import pandas as pd
import pytest
from backend.api_models import PredictionRequest
from backend.feature_extractor import FeatureExtractor
from model_pipeline.data_processor import DataProcessor

DATA_FILENAME = "data.csv"
ZONES_FILENAME = "zones.csv"
TEST_FILENAME = "test.csv"
OUTPUT_FOLDER = "data"
DATA_FILEPATH = os.path.join(OUTPUT_FOLDER, DATA_FILENAME)
ZONES_FILEPATH = os.path.join(OUTPUT_FOLDER, ZONES_FILENAME)
TEST_FILEPATH = os.path.join(OUTPUT_FOLDER, TEST_FILENAME)

processor = DataProcessor(DATA_FILENAME, ZONES_FILENAME, OUTPUT_FOLDER)


def load_first_200_lines(data_filename: str) -> pd.DataFrame:
    return pd.read_csv(data_filename, nrows=200)


def create_prediction_requests(df: pd.DataFrame) -> list:
    requests = []
    count = 0
    for _, row in df.iterrows():
        request = PredictionRequest(
            trip_id=str(count),
            request_datetime=row["tpep_pickup_datetime"],
            trip_distance=row["trip_distance"],
            PULocationID=row["PULocationID"],
            DOLocationID=row["DOLocationID"],
            Airport=1 if row["Airport_fee"] > 0 else 0,
        )
        count += 1
        requests.append(request)
    return requests


@pytest.fixture(scope="module")
def setup_data():
    df_raw = load_first_200_lines(DATA_FILEPATH)

    feature_extractor = FeatureExtractor(ZONES_FILENAME, OUTPUT_FOLDER, TEST_FILENAME)

    return df_raw, feature_extractor


def test_feature_extraction(setup_data):
    df_raw, feature_extractor = setup_data

    columns_needed = [
        "tpep_pickup_datetime",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "Airport_fee",
    ]
    df_extracted = df_raw[columns_needed].copy()

    # Create prediction requests
    prediction_requests = create_prediction_requests(df_extracted)

    # Extract features for each request
    feature_dfs = []
    for request in prediction_requests:
        feature_df = feature_extractor.extract_features(request)
        feature_dfs.append(feature_df)

    # Combine all feature dataframes into one
    df_features = pd.concat(feature_dfs, ignore_index=True)

    # Load the expected test dataframe created by DataProcessor
    df_zone = pd.read_csv(ZONES_FILEPATH)
    df_expected = processor.extract_features(df_raw, remove_invalid=False)
    df_expected = processor.merge_location_data(df_expected, df_zone)
    df_expected = processor.encode_categorical(df_expected)
    df_expected = df_expected.astype(float)

    # Remove target trip_time
    df_expected = df_expected.drop(columns=["trip_time"])

    # Ensure the column orders are the same
    df_features = df_features[df_expected.columns]

    df_features.sort_values(by=df_features.columns.tolist(), inplace=True)
    df_expected.sort_values(by=df_expected.columns.tolist(), inplace=True)

    df_features.reset_index(drop=True, inplace=True)
    df_expected.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(
        df_features, df_expected, check_like=False, check_column_type=False
    )


if __name__ == "__main__":
    pytest.main(["-s"])
